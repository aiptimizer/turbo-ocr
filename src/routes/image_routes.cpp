#include "turbo_ocr/routes/image_routes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::decode::NvJpegDecoder;

namespace turbo_ocr::routes {

void register_image_routes(server::WorkPool &pool,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available) {

  // --- /ocr/raw: GPU-direct JPEG decode, Wuffs PNG ---
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    if (req->body().empty()) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return;
    }

    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
      return;
    }

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, &decode, nvjpeg_available, want_layout](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/raw", [&] {
        const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
        size_t len = req->body().size();

        // JPEG with nvJPEG: submit GPU-direct decode + infer as one work item
        if (nvjpeg_available && NvJpegDecoder::is_jpeg(data, len)) {
          auto out = dispatcher.submit([data, len, want_layout](auto &e) {
            thread_local NvJpegDecoder nvjpeg;
            auto [w, h] = nvjpeg.get_dimensions(data, len);
            if (w > 0 && h > 0) {
              auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
              if (nvjpeg.decode_to_gpu(data, len, d_buf, pitch, w, h, e.stream)) {
                turbo_ocr::GpuImage gpu_img{.data = d_buf, .step = pitch, .rows = h, .cols = w};
                try {
                  return e.pipeline->run_with_layout(gpu_img, e.stream, want_layout);
                } catch (const std::exception &) {}
              }
            }
            cv::Mat img = nvjpeg.decode(data, len);
            if (img.empty()) {
              if (len <= static_cast<size_t>(INT_MAX))
                img = cv::imdecode(
                    cv::Mat(1, static_cast<int>(len), CV_8UC1,
                            const_cast<unsigned char *>(data)),
                    cv::IMREAD_COLOR);
            }
            if (img.empty())
              throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
            return e.pipeline->run_with_layout(img, e.stream, want_layout);
          }).get();
          cb(server::json_response(results_to_json(out.results, out.layout)));
          return;
        }

        // Non-JPEG (PNG, etc.) or nvJPEG not available
        cv::Mat img = decode(data, len);
        if (img.empty()) {
          cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
          return;
        }

        auto out = dispatcher.submit([&img, want_layout](auto &e) {
          return e.pipeline->run_with_layout(img, e.stream, want_layout);
        }).get();
        cb(server::json_response(results_to_json(out.results, out.layout)));
      });
    });
  }, {drogon::Post});

  // --- /ocr/batch: nvJPEG batch decode + parallel pipeline ---
  drogon::app().registerHandler(
      "/ocr/batch",
      [&pool, &dispatcher, &decode, nvjpeg_available, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
      return;
    }

    auto json = req->getJsonObject();
    if (!json) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
      return;
    }
    if (!json->isMember("images") || !(*json)["images"].isArray()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Missing images array"));
      return;
    }

    auto &images_json = (*json)["images"];
    size_t n = images_json.size();
    if (n == 0) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BATCH", "Empty images array"));
      return;
    }

    auto b64_strings = std::make_shared<std::vector<std::string>>(n);
    for (size_t i = 0; i < n; ++i)
      (*b64_strings)[i] = images_json[static_cast<int>(i)].asString();

    server::submit_work(pool, std::move(callback),
        [b64_strings, n, &dispatcher, &decode, nvjpeg_available, want_layout](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/batch", [&] {
        auto raw_bytes = std::make_shared<std::vector<std::string>>(n);
        for (size_t i = 0; i < n; ++i)
          (*raw_bytes)[i] = base64_decode((*b64_strings)[i]);

        std::vector<cv::Mat> imgs(n);
        std::vector<size_t> jpeg_indices;
        std::vector<std::pair<const unsigned char *, size_t>> jpeg_buffers;

        if (nvjpeg_available) {
          for (size_t i = 0; i < n; ++i) {
            auto &raw = (*raw_bytes)[i];
            if (raw.size() >= 2 &&
                static_cast<unsigned char>(raw[0]) == 0xFF &&
                static_cast<unsigned char>(raw[1]) == 0xD8) {
              jpeg_indices.push_back(i);
              jpeg_buffers.emplace_back(
                  reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
            }
          }
        }

        if (jpeg_buffers.size() >= 2) {
          thread_local NvJpegDecoder tl_nvjpeg;
          auto batch_mats = tl_nvjpeg.batch_decode(jpeg_buffers);
          for (size_t j = 0; j < jpeg_indices.size(); ++j)
            imgs[jpeg_indices[j]] = std::move(batch_mats[j]);
        }

        for (size_t i = 0; i < n; ++i) {
          if (!imgs[i].empty()) continue;
          auto &raw = (*raw_bytes)[i];
          if (raw.empty()) continue;
          imgs[i] = decode(
              reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
        }

        std::vector<cv::Mat> valid_imgs;
        std::vector<size_t> valid_indices;
        valid_imgs.reserve(n);
        valid_indices.reserve(n);
        for (size_t i = 0; i < n; ++i) {
          if (!imgs[i].empty()) {
            valid_imgs.push_back(std::move(imgs[i]));
            valid_indices.push_back(i);
          }
        }
        if (valid_imgs.empty()) {
          cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "No valid images"));
          return;
        }

        if (want_layout) {
          // Layout path: run_with_layout per image (layout not supported in run_batch)
          struct BatchItem {
            std::vector<OCRResultItem> results;
            std::vector<turbo_ocr::layout::LayoutBox> layout;
          };
          std::vector<BatchItem> all_items(n);

          dispatcher.submit([&](auto &e) {
            for (size_t j = 0; j < valid_imgs.size(); ++j) {
              auto out = e.pipeline->run_with_layout(valid_imgs[j], e.stream, true);
              auto idx = valid_indices[j];
              all_items[idx].results = std::move(out.results);
              all_items[idx].layout = std::move(out.layout);
            }
          }).get();

          std::string json_str;
          json_str.reserve(n * 1024);
          json_str += "{\"batch_results\":[";
          for (size_t i = 0; i < n; ++i) {
            if (i > 0) json_str += ',';
            json_str += results_to_json(all_items[i].results, all_items[i].layout);
          }
          json_str += "]}";
          cb(server::json_response(std::move(json_str)));
        } else {
          // Fast path: run_batch without layout
          constexpr int kMaxBatch = 8;
          std::vector<std::vector<OCRResultItem>> all_results(n);

          dispatcher.submit([&](auto &e) {
            for (size_t offset = 0; offset < valid_imgs.size(); offset += kMaxBatch) {
              size_t end = std::min(offset + kMaxBatch, valid_imgs.size());
              std::vector<cv::Mat> chunk(
                  std::make_move_iterator(valid_imgs.begin() + offset),
                  std::make_move_iterator(valid_imgs.begin() + end));
              auto chunk_results = e.pipeline->run_batch(chunk, e.stream);
              for (size_t j = 0; j < chunk_results.size(); ++j)
                all_results[valid_indices[offset + j]] = std::move(chunk_results[j]);
            }
          }).get();

          std::string json_str;
          json_str.reserve(n * 1024);
          json_str += "{\"batch_results\":[";
          for (size_t i = 0; i < n; ++i) {
            if (i > 0) json_str += ',';
            json_str += results_to_json(all_results[i]);
          }
          json_str += "]}";
          cb(server::json_response(std::move(json_str)));
        }
      });
    });
  }, {drogon::Post});

  // --- /ocr/pixels: raw BGR pixel data, zero decode overhead ---
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&pool, &dispatcher, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
      return;
    }

    auto w_str = req->getHeader("X-Width");
    auto h_str = req->getHeader("X-Height");
    auto c_str = req->getHeader("X-Channels");

    if (w_str.empty() || h_str.empty()) {
      callback(server::error_response(drogon::k400BadRequest, "MISSING_HEADER", "Missing X-Width or X-Height headers"));
      return;
    }

    int width, height, channels;
    try {
      width = std::stoi(w_str);
      height = std::stoi(h_str);
      channels = c_str.empty() ? 3 : std::stoi(c_str);
    } catch (const std::exception &) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_HEADER",
          "Invalid X-Width, X-Height, or X-Channels header value"));
      return;
    }

    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DIMENSIONS", "Invalid dimensions or channels"));
      return;
    }

    constexpr int kMaxPixelDim = 16384;
    if (width > kMaxPixelDim || height > kMaxPixelDim) {
      callback(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
          std::format("Dimensions {}x{} exceed maximum of {}x{}", width, height, kMaxPixelDim, kMaxPixelDim)));
      return;
    }

    size_t expected = static_cast<size_t>(width) * height * channels;
    if (req->body().size() != expected) {
      callback(server::error_response(drogon::k400BadRequest, "BODY_SIZE_MISMATCH",
          std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                      expected, width, height, channels, req->body().size())));
      return;
    }

    server::submit_work(pool, std::move(callback),
        [req, &dispatcher, width, height, channels, want_layout](server::DrogonCallback &cb) {
      server::run_with_error_handling(cb, "/ocr/pixels", [&] {
        cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                    const_cast<char *>(req->body().data()));

        auto out = dispatcher.submit([&img, want_layout](auto &e) {
          return e.pipeline->run_with_layout(img, e.stream, want_layout);
        }).get();
        cb(server::json_response(results_to_json(out.results, out.layout)));
      });
    });
  }, {drogon::Post});
}

} // namespace turbo_ocr::routes
