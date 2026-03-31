#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
GENERATOR="${GENERATOR:-./src/generate_cctag_dataset.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/testing/generated_testing_suites}"
TEST_COUNT="${TEST_COUNT:-1000}"

common_args=(
  --num_images "${TEST_COUNT}"
  --output_size 640x400
  --heatmap_stride 8
  --heatmap_sigma 2.0
  --marker_style cctag_source
  --num_rings 3
)

generate_one() {
  local name="$1"
  shift
  echo "==> generating ${name}"
  "${PYTHON_BIN}" "${GENERATOR}" "$@"
}

mkdir -p "${OUTPUT_ROOT}"

generate_one "testing_small_hard" \
  --output_dir "${OUTPUT_ROOT}/testing_small_hard" \
  --seed 101 \
  --marker_min 40 \
  --marker_max 120 \
  --occ_min 0.85 \
  --occ_max 0.98 \
  --partial_out_prob 0.25 \
  --partial_out_max_ratio 0.25 \
  --empty_negative_ratio 0.15 \
  --boundary_target_ratio 0.20 \
  --degradation_preset soft_focus \
  --soft_focus_strength 0.95 \
  "${common_args[@]}"

generate_one "testing_boundary_hard" \
  --output_dir "${OUTPUT_ROOT}/testing_boundary_hard" \
  --seed 102 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.85 \
  --occ_max 0.98 \
  --partial_out_prob 0.60 \
  --partial_out_max_ratio 0.45 \
  --empty_negative_ratio 0.15 \
  --boundary_target_ratio 0.40 \
  --degradation_preset standard \
  --soft_focus_strength 0.90 \
  "${common_args[@]}"

generate_one "testing_negative_hard" \
  --output_dir "${OUTPUT_ROOT}/testing_negative_hard" \
  --seed 103 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.85 \
  --occ_max 0.98 \
  --partial_out_prob 0.25 \
  --partial_out_max_ratio 0.25 \
  --empty_negative_ratio 0.35 \
  --boundary_target_ratio 0.20 \
  --degradation_preset standard \
  --soft_focus_strength 0.90 \
  "${common_args[@]}"

generate_one "testing_extreme_mix" \
  --output_dir "${OUTPUT_ROOT}/testing_extreme_mix" \
  --seed 104 \
  --marker_min 40 \
  --marker_max 180 \
  --occ_min 0.92 \
  --occ_max 0.995 \
  --partial_out_prob 0.65 \
  --partial_out_max_ratio 0.45 \
  --empty_negative_ratio 0.25 \
  --boundary_target_ratio 0.35 \
  --degradation_preset soft_focus \
  --soft_focus_strength 1.0 \
  "${common_args[@]}"

echo "==> testing suites ready under ${OUTPUT_ROOT}"
