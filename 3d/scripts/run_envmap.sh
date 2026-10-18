#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${ROOT_DIR}/src/optimize.py" \
  --case envmap --backend reference --npoints 512 \
  --output "${ROOT_DIR}/data/optimized/envmap/points.npz" "$@"
