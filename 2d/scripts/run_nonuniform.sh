#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET="image"
IMAGE_PATH="${ROOT_DIR}/data/targets/empty_play_room_1k.exr"

# paper uses instances=16
cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/src/run_nonuniform.py"
  --target "${TARGET}"
  --outdir "${ROOT_DIR}/data/optimized/${TARGET}"
  --ns 4 8 16 32 64 128 256 512 1024
  --instances 4
  --device cuda
  --nregions 4096
  --nspace_samples 262144
  --nepochs 1000
  --nepochs_params 500
  --lr_init 1e-4
  --lr_final 1e-7
  --tau0 20
  --tau1 2000
  --uv_boundary_mode clamp
  --pointset_selection best
  --clamp_eps 1e-12
  --gt_nregions 32768
  --gt_nspace_samples 262144
  --areas_batch_size 8192
  --areas_space_batch_size 131072
  --region_chunk_size 8192 \
  --gt_reps 8
  --seed 0
  --show_progress
)

if [[ "${TARGET}" == "image" ]]; then
  cmd+=(--image "${IMAGE_PATH}")
fi

"${cmd[@]}"
