#pragma once

#include <cstring>
#include <format>
#include <future>
#include <iostream>
#include <mutex>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#ifndef USE_CPU_ONLY
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"
#include "ocr.grpc.pb.h"

namespace turbo_ocr::server {

enum class GrpcResponseMode { json_bytes, structured };

inline cv::Mat grpc_decode_image(std::string_view image_data) {
  auto *data = reinterpret_cast<const unsigned char *>(image_data.data());
  auto len = image_data.size();
  if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
#ifndef USE_CPU_ONLY
    thread_local decode::NvJpegDecoder tl_nvjpeg;
    if (tl_nvjpeg.available()) {
      cv::Mat img = tl_nvjpeg.decode(data, len);
      if (!img.empty()) return img;
    }
#endif
    if (len > static_cast<size_t>(INT_MAX)) return {};
    return cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                                const_cast<unsigned char *>(data)),
                        cv::IMREAD_COLOR);
  }
  if (decode::FastPngDecoder::is_png(data, len))
    return decode::FastPngDecoder::decode(data, len);
  return {};
}

class OCRServiceImpl final : public ocr::OCRService::Service {
public:
#ifndef USE_CPU_ONLY
  OCRServiceImpl(pipeline::PipelineDispatcher &dispatcher,
                 GrpcResponseMode mode,
                 render::PdfRenderer *pdf_renderer = nullptr,
                 pdf::PdfMode default_pdf_mode = pdf::PdfMode::Ocr,
                 bool layout_available = false)
      : dispatcher_(&dispatcher),
        mode_(mode),
        pdf_renderer_(pdf_renderer),
        default_pdf_mode_(default_pdf_mode),
        layout_available_(layout_available) {}
#endif

  /// CPU-friendly constructor: takes an InferFunc instead of a dispatcher.
  OCRServiceImpl(InferFunc infer_fn,
                 GrpcResponseMode mode,
                 render::PdfRenderer *pdf_renderer = nullptr,
                 pdf::PdfMode default_pdf_mode = pdf::PdfMode::Ocr,
                 bool layout_available = false)
      : infer_fn_(std::move(infer_fn)),
        mode_(mode),
        pdf_renderer_(pdf_renderer),
        default_pdf_mode_(default_pdf_mode),
        layout_available_(layout_available) {}

  // ---- Health ----
  grpc::Status Health(grpc::ServerContext *,
                      const ocr::HealthRequest *,
                      ocr::HealthResponse *response) override {
    response->set_status("ok");
    return grpc::Status::OK;
  }

  // ---- Recognize (single image + pixels + layout) ----
  grpc::Status Recognize(grpc::ServerContext *,
                         const ocr::OCRRequest *request,
                         ocr::OCRResponse *response) override {
    bool want_layout = request->layout() && layout_available_;

    // Pixels path: raw BGR pixel data
    if (!request->pixels().empty()) {
      int width = request->width();
      int height = request->height();
      int channels = request->channels();
      if (channels == 0) channels = 3;

      if (width <= 0 || height <= 0 || (channels != 1 && channels != 3))
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            "Invalid dimensions or channels for pixels input");

      constexpr int kMaxPixelDim = 16384;
      if (width > kMaxPixelDim || height > kMaxPixelDim)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Dimensions {}x{} exceed maximum of {}x{}",
                        width, height, kMaxPixelDim, kMaxPixelDim));

      size_t expected = static_cast<size_t>(width) * height * channels;
      if (request->pixels().size() != expected)
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
            std::format("Pixels size mismatch: expected {} bytes ({}x{}x{}), got {}",
                        expected, width, height, channels, request->pixels().size()));

      cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                  const_cast<char *>(request->pixels().data()));

      try {
        auto out = run_infer(img, want_layout);
        fill_response(response, out.results, out.layout);
        return grpc::Status::OK;
      } catch (const turbo_ocr::PoolExhaustedError &e) {
        return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, e.what());
      } catch (const std::exception &e) {
        std::cerr << std::format("[gRPC] Pixels inference error: {}\n", e.what());
        return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
      }
    }

    // Image path: encoded image bytes
    if (request->image().empty()) [[unlikely]]
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Empty image");

    cv::Mat img = grpc_decode_image(request->image());
    if (img.empty()) [[unlikely]]
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Decode failed");

    try {
      auto out = run_infer(img, want_layout);
      fill_response(response, out.results, out.layout);
      return grpc::Status::OK;
    } catch (const turbo_ocr::PoolExhaustedError &e) {
      return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, e.what());
    } catch (const std::exception &e) {
      std::cerr << std::format("[gRPC] Inference error: {}\n", e.what());
      return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
    } catch (...) {
      std::cerr << "[gRPC] Inference error: unknown exception\n";
      return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
    }
  }

  // ---- RecognizeBatch ----
  grpc::Status RecognizeBatch(grpc::ServerContext *,
                              const ocr::OCRBatchRequest *request,
                              ocr::OCRBatchResponse *response) override {
    int n = request->images_size();
    if (n == 0)
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Empty images array");

    bool want_layout = request->layout() && layout_available_;

    // Decode all images first
    std::vector<cv::Mat> imgs(n);
    for (int i = 0; i < n; ++i) {
      imgs[i] = grpc_decode_image(request->images(i));
    }

    // Check we have at least one valid image
    bool any_valid = false;
    for (int i = 0; i < n; ++i) {
      if (!imgs[i].empty()) { any_valid = true; break; }
    }
    if (!any_valid)
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "No valid images");

    // Dispatch each image through the pipeline
    response->set_total_images(n);
    for (int i = 0; i < n; ++i) {
      auto *batch_entry = response->add_batch_results();
      if (imgs[i].empty()) {
        batch_entry->set_num_detections(0);
        continue;
      }
      try {
        auto out = run_infer(imgs[i], want_layout);
        fill_response(batch_entry, out.results, out.layout);
      } catch (const std::exception &e) {
        std::cerr << std::format("[gRPC Batch] Image {} error: {}\n", i, e.what());
        batch_entry->set_num_detections(0);
      }
    }

    return grpc::Status::OK;
  }

  // ---- RecognizePDF ----
  grpc::Status RecognizePDF(grpc::ServerContext *,
                            const ocr::OCRPDFRequest *request,
                            ocr::OCRPDFResponse *response) override {
    if (!pdf_renderer_)
      return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                          "PDF rendering not available on this server");

    if (request->pdf_data().empty())
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Empty PDF data");

    const auto *pdf_data = reinterpret_cast<const uint8_t *>(request->pdf_data().data());
    size_t pdf_len = request->pdf_data().size();

    bool want_layout = request->layout() && layout_available_;

    int dpi = request->dpi();
    if (dpi == 0) dpi = 100;
    if (dpi < 50 || dpi > 600)
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "DPI must be between 50 and 600");

    pdf::PdfMode req_mode = default_pdf_mode_;
    if (!request->mode().empty())
      req_mode = pdf::parse_pdf_mode(request->mode(), default_pdf_mode_);

    // Open PDF for text-layer modes
    std::unique_ptr<pdf::PdfDocument> pdf_doc;
    std::vector<pdf::PdfPageText> page_text_cache;
    if (req_mode != pdf::PdfMode::Ocr) {
      pdf_doc = std::make_unique<pdf::PdfDocument>(pdf_data, pdf_len);
      if (!pdf_doc->ok()) {
        std::cerr << "[gRPC PDF] failed to open PDF for text-layer lookup; "
                     "falling back to mode=ocr\n";
        req_mode = pdf::PdfMode::Ocr;
        pdf_doc.reset();
      } else {
        int np = pdf_doc->page_count();
        page_text_cache.reserve(static_cast<size_t>(std::max(0, np)));
        for (int p = 0; p < np; ++p)
          page_text_cache.push_back(pdf_doc->extract_page(p));
      }
    }

    // Per-page result accumulator
    std::mutex results_mutex;
    struct PdfPageResult {
      std::vector<OCRResultItem> results;
      std::vector<layout::LayoutBox> layout;
      int width = 0, height = 0, effective_dpi = 0;
      pdf::PdfMode resolved_mode = pdf::PdfMode::Ocr;
      std::string_view text_layer_quality = "absent";
    };
    std::vector<PdfPageResult> page_results;

    // Fill results from text layer (PDF points)
    auto fill_from_text_layer_pt =
        [](PdfPageResult &pg, const pdf::PdfPageText &text) {
      pg.width  = static_cast<int>(std::round(text.page_width_pt));
      pg.height = static_cast<int>(std::round(text.page_height_pt));
      pg.effective_dpi = 72;
      pg.results.reserve(text.lines.size());
      for (const auto &line : text.lines) {
        OCRResultItem item;
        item.source = "pdf";
        item.confidence = 1.0f;
        item.text = line.text;
        int ix0 = static_cast<int>(std::round(line.x0_pt));
        int iy0 = static_cast<int>(std::round(line.y0_pt));
        int ix1 = static_cast<int>(std::round(line.x1_pt));
        int iy1 = static_cast<int>(std::round(line.y1_pt));
        item.box[0] = {ix0, iy0};
        item.box[1] = {ix1, iy0};
        item.box[2] = {ix1, iy1};
        item.box[3] = {ix0, iy1};
        pg.results.push_back(std::move(item));
      }
    };

    auto text_layer_quality_for =
        [](const pdf::PdfPageText &text) -> std::string_view {
      if (text.char_count == 0)         return "absent";
      if (text.rotation_deg != 0)       return "rejected";
      if (text.char_count < 10)         return "absent";
      if (text.fffd_count * 20 > text.char_count)     return "rejected";
      if (text.nonprint_count * 10 > text.char_count) return "rejected";
      if (text.lines.empty())           return "absent";
      return "trusted";
    };

    // Pre-populate pages that don't need rendering
    std::vector<uint8_t> need_render;
    bool any_need_render = (req_mode == pdf::PdfMode::Ocr);

    if (req_mode != pdf::PdfMode::Ocr) {
      int np = pdf_doc ? pdf_doc->page_count() : 0;
      page_results.resize(static_cast<size_t>(np));
      need_render.assign(static_cast<size_t>(np), 0);

      for (int p = 0; p < np; ++p) {
        const auto &text = page_text_cache[static_cast<size_t>(p)];
        auto &pg = page_results[static_cast<size_t>(p)];
        pg.text_layer_quality = text_layer_quality_for(text);
        bool has_good_layer = (pg.text_layer_quality == "trusted");

        switch (req_mode) {
          case pdf::PdfMode::Geometric:
            pg.resolved_mode = pdf::PdfMode::Geometric;
            if (has_good_layer) fill_from_text_layer_pt(pg, text);
            else {
              pg.width = static_cast<int>(std::round(text.page_width_pt));
              pg.height = static_cast<int>(std::round(text.page_height_pt));
              pg.effective_dpi = 72;
            }
            if (want_layout) {
              need_render[static_cast<size_t>(p)] = 1;
              any_need_render = true;
            }
            break;
          case pdf::PdfMode::Auto:
            if (has_good_layer) {
              pg.resolved_mode = pdf::PdfMode::Geometric;
              fill_from_text_layer_pt(pg, text);
              if (want_layout) {
                need_render[static_cast<size_t>(p)] = 1;
                any_need_render = true;
              }
            } else {
              pg.resolved_mode = pdf::PdfMode::Ocr;
              need_render[static_cast<size_t>(p)] = 1;
              any_need_render = true;
            }
            break;
          case pdf::PdfMode::AutoVerified:
            pg.resolved_mode = pdf::PdfMode::AutoVerified;
            need_render[static_cast<size_t>(p)] = 1;
            any_need_render = true;
            break;
          default: break;
        }
      }
    }

    // Streamed render + OCR
    std::mutex futures_mutex;
    std::vector<std::future<void>> page_futures;
    render::PdfRenderer::StreamHandle stream_handle;
    int num_pages = 0;

    if (any_need_render) {
      try {
        stream_handle = pdf_renderer_->render_streamed(pdf_data, pdf_len, dpi,
            [&](int page_idx, std::string ppm_path) {
              {
                std::lock_guard<std::mutex> rlock(results_mutex);
                if (page_idx >= static_cast<int>(page_results.size())) {
                  page_results.resize(page_idx + 1);
                  if (req_mode != pdf::PdfMode::Ocr &&
                      page_idx >= static_cast<int>(need_render.size()))
                    need_render.resize(page_idx + 1, 1);
                }
                if (req_mode != pdf::PdfMode::Ocr &&
                    page_idx < static_cast<int>(need_render.size()) &&
                    !need_render[page_idx])
                  return;
              }

              auto fut = std::async(std::launch::async,
                  [&, page_idx, path = std::move(ppm_path)]() {
                cv::Mat img = render::PdfRenderer::decode_ppm(path);
                if (img.empty()) {
                  std::cerr << std::format("[gRPC PDF] Failed to decode PPM for page {}\n", page_idx);
                  return;
                }
                int pw = img.cols, ph = img.rows;

                pdf::PdfMode page_mode;
                {
                  std::lock_guard<std::mutex> rlock(results_mutex);
                  page_mode = (page_idx < static_cast<int>(page_results.size()))
                      ? page_results[page_idx].resolved_mode
                      : pdf::PdfMode::Ocr;
                }

                std::vector<OCRResultItem> rec_results;
                std::vector<layout::LayoutBox> layout_snapshot;

                // Geometric mode with layout: run full inference to get layout
                // (CPU has no run_layout_only; GPU path also benefits from unified code)
                if (page_mode == pdf::PdfMode::Geometric && want_layout) {
                  auto infer_out = run_infer(img, true);
                  layout_snapshot = std::move(infer_out.layout);
                } else if (page_mode != pdf::PdfMode::Geometric) {
                  auto infer_out = run_infer(img, want_layout);
                  rec_results = std::move(infer_out.results);
                  layout_snapshot = std::move(infer_out.layout);
                  for (auto &it : rec_results) it.source = "ocr";
                }

                if (page_mode == pdf::PdfMode::AutoVerified &&
                    page_idx < static_cast<int>(page_text_cache.size()) && pdf_doc) {
                  for (auto &item : rec_results) {
                    const float px_to_pt = 72.0f / static_cast<float>(dpi);
                    auto [ix0, iy0, ix1, iy1] = turbo_ocr::aabb(item.box);
                    float x0 = ix0 * px_to_pt, y0 = iy0 * px_to_pt;
                    float x1 = ix1 * px_to_pt, y1 = iy1 * px_to_pt;
                    std::string native =
                        pdf_doc->text_in_rect_pt(page_idx, x0, y0, x1, y1);
                    auto verdict = pdf::passes_sanity_check(
                        native, x1 - x0, y1 - y0);
                    if (verdict.accept) {
                      item.text = std::move(native);
                      item.source = "pdf";
                      item.confidence = 1.0f;
                    }
                  }
                }

                std::lock_guard<std::mutex> rlock(results_mutex);
                auto &slot = page_results[page_idx];
                if (page_mode == pdf::PdfMode::Geometric) {
                  const float pt_to_px = static_cast<float>(dpi) / 72.0f;
                  for (auto &item : slot.results) {
                    for (int k = 0; k < 4; ++k) {
                      item.box[k][0] = static_cast<int>(
                          std::round(item.box[k][0] * pt_to_px));
                      item.box[k][1] = static_cast<int>(
                          std::round(item.box[k][1] * pt_to_px));
                    }
                  }
                } else {
                  slot.results = std::move(rec_results);
                }
                slot.layout        = std::move(layout_snapshot);
                slot.width         = pw;
                slot.height        = ph;
                slot.effective_dpi = dpi;
                if (page_mode == pdf::PdfMode::Ocr)
                  slot.resolved_mode = pdf::PdfMode::Ocr;
              });

              std::lock_guard lock(futures_mutex);
              page_futures.push_back(std::move(fut));
            });
        num_pages = stream_handle.num_pages;
      } catch (const std::exception &e) {
        for (auto &f : page_futures) { try { f.get(); } catch (...) {} }
        std::cerr << std::format("[gRPC PDF] PDF render failed: {}\n", e.what());
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "PDF render failed");
      }
    } else {
      num_pages = pdf_doc ? pdf_doc->page_count() : 0;
    }

    {
      std::lock_guard<std::mutex> rlock(results_mutex);
      if (static_cast<int>(page_results.size()) < num_pages)
        page_results.resize(num_pages);
    }

    for (auto &f : page_futures) {
      try { f.get(); } catch (const std::exception &e) {
        std::cerr << std::format("[gRPC PDF] page error: {}\n", e.what());
      }
    }

    if (num_pages == 0)
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "PDF contains no pages");

    // Build response
    for (int i = 0; i < num_pages; ++i) {
      auto *page = response->add_pages();
      auto &pg = page_results[static_cast<size_t>(i)];
      page->set_page_number(i + 1);
      page->set_width(pg.width);
      page->set_height(pg.height);
      page->set_dpi(pg.effective_dpi > 0 ? pg.effective_dpi : dpi);
      page->set_mode(std::string(pdf::mode_name(pg.resolved_mode)));
      page->set_text_layer_quality(std::string(pg.text_layer_quality));

      if (mode_ == GrpcResponseMode::json_bytes) {
        page->set_json_response(results_to_json(pg.results, pg.layout));
      } else {
        fill_page_results(page, pg.results);
      }
    }

    return grpc::Status::OK;
  }

private:
  void fill_response(ocr::OCRResponse *response,
                     std::vector<OCRResultItem> &results,
                     std::vector<layout::LayoutBox> &layout_boxes) {
    response->set_num_detections(static_cast<int>(results.size()));
    if (mode_ == GrpcResponseMode::json_bytes) {
      if (layout_boxes.empty())
        response->set_json_response(results_to_json(results));
      else
        response->set_json_response(results_to_json(results, layout_boxes));
    } else {
      response->mutable_results()->Reserve(static_cast<int>(results.size()));
      for (const auto &item : results) {
        auto *result = response->add_results();
        result->set_text(item.text);
        result->set_confidence(item.confidence);
        result->mutable_bounding_box()->Reserve(4);
        for (int k = 0; k < 4; ++k) {
          auto *bbox = result->add_bounding_box();
          bbox->mutable_x()->Reserve(1);
          bbox->mutable_y()->Reserve(1);
          bbox->add_x(static_cast<float>(item.box[k][0]));
          bbox->add_y(static_cast<float>(item.box[k][1]));
        }
      }
    }
  }

  void fill_page_results(ocr::OCRPageResult *page,
                         const std::vector<OCRResultItem> &results) {
    page->mutable_results()->Reserve(static_cast<int>(results.size()));
    for (const auto &item : results) {
      auto *result = page->add_results();
      result->set_text(item.text);
      result->set_confidence(item.confidence);
      result->mutable_bounding_box()->Reserve(4);
      for (int k = 0; k < 4; ++k) {
        auto *bbox = result->add_bounding_box();
        bbox->mutable_x()->Reserve(1);
        bbox->mutable_y()->Reserve(1);
        bbox->add_x(static_cast<float>(item.box[k][0]));
        bbox->add_y(static_cast<float>(item.box[k][1]));
      }
    }
  }

  /// Unified inference: uses InferFunc if set, otherwise dispatcher.
  pipeline::OcrPipelineResult run_infer(const cv::Mat &img, bool want_layout) {
    if (infer_fn_) {
      auto r = infer_fn_(img, want_layout);
      return pipeline::OcrPipelineResult{
          .results = std::move(r.results),
          .layout  = std::move(r.layout),
      };
    }
#ifndef USE_CPU_ONLY
    return dispatcher_->submit([&img, want_layout](auto &e) {
      return e.pipeline->run_with_layout(img, e.stream, want_layout);
    }).get();
#else
    throw std::logic_error("No inference backend configured");
#endif
  }

#ifndef USE_CPU_ONLY
  pipeline::PipelineDispatcher *dispatcher_ = nullptr;
#endif
  InferFunc infer_fn_;
  GrpcResponseMode mode_;
  render::PdfRenderer *pdf_renderer_ = nullptr;
  pdf::PdfMode default_pdf_mode_ = pdf::PdfMode::Ocr;
  bool layout_available_ = false;
};

/// Start gRPC server on a background thread. Returns the server and thread.
/// Caller must keep both alive. Call server->Shutdown() to stop.
struct GrpcHandle {
  std::unique_ptr<grpc::Server> server;
  std::jthread thread;
};

namespace detail {

inline GrpcHandle launch_grpc_server(std::shared_ptr<OCRServiceImpl> service,
                                      int port) {
  int kMaxMsg = 100 * 1024 * 1024;
  if (const char *env = std::getenv("MAX_BODY_BYTES"))
    kMaxMsg = std::max(1, std::atoi(env));
  int cqs = 10;
  if (const char *env = std::getenv("GRPC_CQS"))
    cqs = std::max(1, std::atoi(env));

  auto address = std::format("0.0.0.0:{}", port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.SetMaxReceiveMessageSize(kMaxMsg);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, cqs * 2);
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 1);
  builder.AddChannelArgument(GRPC_ARG_MINIMAL_STACK, 1);

  auto server = builder.BuildAndStart();
  std::cout << std::format("gRPC server listening on {}\n", address);

  auto thread = std::jthread([srv = server.get(), svc = std::move(service)]() {
    srv->Wait();
  });

  return {std::move(server), std::move(thread)};
}

} // namespace detail

#ifndef USE_CPU_ONLY
/// Start gRPC server using a PipelineDispatcher (GPU path).
inline GrpcHandle start_grpc_server(pipeline::PipelineDispatcher &dispatcher,
                                     int port,
                                     render::PdfRenderer *pdf_renderer = nullptr,
                                     pdf::PdfMode default_pdf_mode = pdf::PdfMode::Ocr,
                                     bool layout_available = false) {
  auto mode = GrpcResponseMode::json_bytes;
  if (const char *env = std::getenv("GRPC_RESPONSE_MODE")) {
    if (std::strcmp(env, "structured") == 0)
      mode = GrpcResponseMode::structured;
  }

  auto service = std::make_shared<OCRServiceImpl>(
      dispatcher, mode, pdf_renderer, default_pdf_mode, layout_available);
  return detail::launch_grpc_server(std::move(service), port);
}
#endif

/// Start gRPC server using an InferFunc (CPU path, also usable from GPU).
inline GrpcHandle start_grpc_server(InferFunc infer_fn,
                                     int port,
                                     render::PdfRenderer *pdf_renderer = nullptr,
                                     pdf::PdfMode default_pdf_mode = pdf::PdfMode::Ocr,
                                     bool layout_available = false) {
  auto mode = GrpcResponseMode::json_bytes;
  if (const char *env = std::getenv("GRPC_RESPONSE_MODE")) {
    if (std::strcmp(env, "structured") == 0)
      mode = GrpcResponseMode::structured;
  }

  auto service = std::make_shared<OCRServiceImpl>(
      std::move(infer_fn), mode, pdf_renderer, default_pdf_mode, layout_available);
  return detail::launch_grpc_server(std::move(service), port);
}

} // namespace turbo_ocr::server
