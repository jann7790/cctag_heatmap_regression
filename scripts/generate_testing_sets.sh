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
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/testing}"

SMALL_HARD_COUNT="${SMALL_HARD_COUNT:-1000}"
BOUNDARY_HARD_COUNT="${BOUNDARY_HARD_COUNT:-1000}"
NEGATIVE_HARD_COUNT="${NEGATIVE_HARD_COUNT:-1000}"
EXTREME_MIX_COUNT="${EXTREME_MIX_COUNT:-1000}"
OVEREXPOSURE_HARD_COUNT="${OVEREXPOSURE_HARD_COUNT:-1000}"

common_args=(
  --output_size 640x400
  --marker_style cctag_source
  --num_rings 3
  --heatmap_stride 4
  --heatmap_sigma 2.0
  --occlusion_style aggressive
)

ensure_absent() {
  local target="$1"
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing path: ${target}" >&2
    exit 1
  fi
}

generate_one() {
  local name="$1"
  shift
  echo "==> generating ${name}"
  "${PYTHON_BIN}" "${GENERATOR}" "$@"
}

mkdir -p "${OUTPUT_ROOT}"

SMALL_HARD_DIR="${OUTPUT_ROOT}/testing_small_hard"
BOUNDARY_HARD_DIR="${OUTPUT_ROOT}/testing_boundary_hard"
NEGATIVE_HARD_DIR="${OUTPUT_ROOT}/testing_negative_hard"
EXTREME_MIX_DIR="${OUTPUT_ROOT}/testing_extreme_mix"
OVEREXPOSURE_HARD_DIR="${OUTPUT_ROOT}/testing_overexposure_hard"

ensure_absent "${SMALL_HARD_DIR}"
ensure_absent "${BOUNDARY_HARD_DIR}"
ensure_absent "${NEGATIVE_HARD_DIR}"
ensure_absent "${EXTREME_MIX_DIR}"
ensure_absent "${OVEREXPOSURE_HARD_DIR}"

generate_one "testing_small_hard" \
  --num_images "${SMALL_HARD_COUNT}" \
  --output_dir "${SMALL_HARD_DIR}" \
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
  --num_images "${BOUNDARY_HARD_COUNT}" \
  --output_dir "${BOUNDARY_HARD_DIR}" \
  --seed 102 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.85 \
  --occ_max 0.98 \
  --partial_out_prob 0.60 \
  --partial_out_max_ratio 0.45 \
  --empty_negative_ratio 0.15 \
  --boundary_target_ratio 0.40 \
  --degradation_preset soft_focus \
  --soft_focus_strength 0.90 \
  "${common_args[@]}"

generate_one "testing_negative_hard" \
  --num_images "${NEGATIVE_HARD_COUNT}" \
  --output_dir "${NEGATIVE_HARD_DIR}" \
  --seed 103 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.85 \
  --occ_max 0.98 \
  --partial_out_prob 0.25 \
  --partial_out_max_ratio 0.25 \
  --empty_negative_ratio 0.35 \
  --boundary_target_ratio 0.20 \
  --degradation_preset soft_focus \
  --soft_focus_strength 0.90 \
  --background_complexity complex \
  "${common_args[@]}"

generate_one "testing_extreme_mix" \
  --num_images "${EXTREME_MIX_COUNT}" \
  --output_dir "${EXTREME_MIX_DIR}" \
  --seed 104 \
  --marker_min 40 \
  --marker_max 180 \
  --occ_min 0.92 \
  --occ_max 0.995 \
  --occ_distribution beta_high \
  --partial_out_prob 0.65 \
  --partial_out_max_ratio 0.45 \
  --empty_negative_ratio 0.20 \
  --boundary_target_ratio 0.35 \
  --degradation_preset soft_focus \
  --soft_focus_strength 0.98 \
  --background_complexity complex \
  --overexposure_prob 0.25 \
  "${common_args[@]}"

generate_one "testing_overexposure_hard" \
  --num_images "${OVEREXPOSURE_HARD_COUNT}" \
  --output_dir "${OVEREXPOSURE_HARD_DIR}" \
  --seed 105 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.20 \
  --occ_max 0.75 \
  --partial_out_prob 0.20 \
  --partial_out_max_ratio 0.30 \
  --empty_negative_ratio 0.30 \
  --boundary_target_ratio 0.20 \
  --background_complexity complex \
  --overexposure_prob 0.85 \
  --blur_min 1 \
  --blur_max 4 \
  --noise_std_min 2 \
  --noise_std_max 12 \
  --brightness_min 10 \
  --brightness_max 45 \
  --contrast_min 0.65 \
  --contrast_max 1.15 \
  --motion_blur_prob 0.25 \
  --scintillation_prob 0.20 \
  --soft_focus_strength 0.45 \
  "${common_args[@]}"

cat <<EOF
==> testing suites ready under ${OUTPUT_ROOT}

Benchmark example:
  uv run python src/benchmark.py --runs_dir outputs/runs --suites_dir ${OUTPUT_ROOT}
EOF
