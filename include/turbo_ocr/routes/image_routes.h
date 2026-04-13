#pragma once

#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#include "turbo_ocr/server/server_types.h"

namespace turbo_ocr::routes {

/// Register /ocr/raw, /ocr/batch, /ocr/pixels routes (GPU paths).
void register_image_routes(server::WorkPool &pool,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available);

} // namespace turbo_ocr::routes
