#include "turbo_ocr/routes/common_routes.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

namespace turbo_ocr::routes {

void register_health_route() {
  auto health_ok = [](const drogon::HttpRequestPtr &,
                      std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
    callback(server::make_response(drogon::k200OK, "ok"));
  };
  drogon::app().registerHandler("/health", health_ok, {drogon::Get});
  drogon::app().registerHandler("/health/live", health_ok, {drogon::Get});
  drogon::app().registerHandler("/health/ready", health_ok, {drogon::Get});
}

void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::ImageDecoder &decode,
                                bool layout_available) {
  drogon::app().registerHandler(
      "/ocr",
      [&pool, &infer, &decode, layout_available](
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
        if (!json->isMember("image") || !(*json)["image"].isString()
            || (*json)["image"].asString().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "MISSING_IMAGE", "Empty or missing image field"));
          return;
        }

        auto b64_str = std::make_shared<std::string>((*json)["image"].asString());

        server::submit_work(pool, std::move(callback),
            [b64_str, &infer, &decode, want_layout](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr", [&] {
            std::string decoded_bytes = turbo_ocr::base64_decode(*b64_str);
            if (decoded_bytes.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64"));
              return;
            }

            cv::Mat img = decode(
                reinterpret_cast<const unsigned char *>(decoded_bytes.data()),
                decoded_bytes.size());
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }

            auto inf = infer(img, want_layout);
            cb(server::json_response(turbo_ocr::results_to_json(inf.results, inf.layout)));
          });
        });
      },
      {drogon::Post});
}

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available) {
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &infer, &decode, layout_available](
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
            [req, &infer, &decode, want_layout](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/raw", [&] {
            const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();

            cv::Mat img = decode(data, len);
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }

            auto inf = infer(img, want_layout);
            cb(server::json_response(turbo_ocr::results_to_json(inf.results, inf.layout)));
          });
        });
      },
      {drogon::Post});
}

void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available) {
  register_health_route();
  register_ocr_base64_route(pool, infer, decode, layout_available);
  register_ocr_raw_route(pool, infer, decode, layout_available);
}

} // namespace turbo_ocr::routes
