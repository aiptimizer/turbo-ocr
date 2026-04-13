#!/bin/bash
set -euo pipefail

# Start nginx reverse proxy (absorbs connection storms, keep-alive to Drogon)
nginx -c /app/docker/nginx.conf

# Drop to non-root user and run the OCR server
# TRT engines are auto-built from ONNX on first startup (cached by TRT version + model hash)
exec gosu ocr "$@"
