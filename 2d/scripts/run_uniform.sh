#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# paper uses instances=16
"${PYTHON_BIN}" "${ROOT_DIR}/src/run_uniform.py" \
  --target uniform \
  --outdir "${ROOT_DIR}/data/optimized/uniform" \
  --ns 4 8 16 32 64 128 256 512 1024 \
  --instances 4 \
  --device cuda \
  --nregions 8192 \
  --nepochs 2001 \
  --nepochs_params 1001 \
  --lr_init 1e-3 \
  --lr_final 1e-7 \
  --tau0 20 \
  --tau1 2000 \
  --clamp_eps 1e-12 \
  --pair_batch_size 4096 \
  --point_chunk_size 1024 \
  --region_chunk_size 8192 \
  --seed 0 \
  --show_progress
