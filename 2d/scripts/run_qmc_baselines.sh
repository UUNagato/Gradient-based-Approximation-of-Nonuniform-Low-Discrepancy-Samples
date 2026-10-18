#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET="image"
IMAGE_PATH="${ROOT_DIR}/data/targets/empty_play_room_1k.exr"

# paper uses instances=128
cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/src/run_qmc_baselines.py"
  --target "${TARGET}"
  --outdir "${ROOT_DIR}/data/baselines/${TARGET}"
  --ns 4 8 16 32 64 128 256 512 1024
  --methods random_base sobol_base sobol_shifted sobol_scrambled halton_base halton_shifted halton_scrambled
  --instances 4
  --device cuda
  --scramble LMS_DS
  --seed 0
  --gt_nregions 32768
  --gt_nspace_samples 262144
  --areas_batch_size 8192
  --areas_space_batch_size 131072
  --region_chunk_size 8192 \
  --pair_batch_size 4096 \
  --gt_reps 8 \
  --save_points \
  --show_progress
)

if [[ "${TARGET}" == "image" ]]; then
  cmd+=(--image "${IMAGE_PATH}")
fi

"${cmd[@]}"
