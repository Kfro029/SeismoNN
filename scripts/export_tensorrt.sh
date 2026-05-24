#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-outputs/cnn_baseline/model.onnx}"
ENGINE_PATH="${2:-outputs/cnn_baseline/model.engine}"

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --minShapes=features:1x2x1723x501 \
  --optShapes=features:1x2x1723x501 \
  --maxShapes=features:1x2x1723x501
