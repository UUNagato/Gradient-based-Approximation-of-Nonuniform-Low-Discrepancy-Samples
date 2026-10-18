#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${ROOT_DIR}/src/train_neural.py" \
  --case brdf --output-dir "${ROOT_DIR}/data/optimized/neural" "$@"
