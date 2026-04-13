#pragma once

#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::routes {

/// Register /health, /ocr (base64 JSON), /ocr/raw (CPU decode path).
/// The GPU binary overrides /ocr/raw with its own nvJPEG version in
/// image_routes, so register_common_routes is only used by cpu_main.
/// readiness_check: optional callable that returns true if the server is ready.
/// Used by /health/ready to verify GPU/pipeline is responsive.
void register_health_route(std::function<bool()> readiness_check = nullptr);

void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::ImageDecoder &decode,
                                bool layout_available);

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available);

/// Convenience: register /health + /ocr + /ocr/raw (CPU paths).
void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available);

} // namespace turbo_ocr::routes
