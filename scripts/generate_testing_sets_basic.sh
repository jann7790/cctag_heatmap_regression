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
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/testing/basic_suites}"

BASE_COUNT="${BASE_COUNT:-1000}"
HARD_COUNT="${HARD_COUNT:-1000}"
EXTREME_COUNT="${EXTREME_COUNT:-1000}"
SMALL_COUNT="${SMALL_COUNT:-1000}"

common_args=(
  --output_size 640x400
  --marker_style cctag_source
  --num_rings 3
  --heatmap_stride 4
  --heatmap_sigma 2.0
  --partial_out_prob 0.25
  --partial_out_max_ratio 0.25
  --empty_negative_ratio 0.15
  --boundary_target_ratio 0.20
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

BASE_DIR="${OUTPUT_ROOT}/testing_base"
HARD_DIR="${OUTPUT_ROOT}/testing_hard"
EXTREME_DIR="${OUTPUT_ROOT}/testing_extreme"
SMALL_DIR="${OUTPUT_ROOT}/testing_small"

ensure_absent "${BASE_DIR}"
ensure_absent "${HARD_DIR}"
ensure_absent "${EXTREME_DIR}"
ensure_absent "${SMALL_DIR}"

generate_one "testing_base" \
  --num_images "${BASE_COUNT}" \
  --output_dir "${BASE_DIR}" \
  --seed 201 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.05 \
  --occ_max 0.50 \
  --soft_focus_strength 0.25 \
  --blur_min 0 --blur_max 3 \
  --noise_std_min 0 --noise_std_max 12 \
  --brightness_min -25 --brightness_max 25 \
  --contrast_min 0.75 --contrast_max 1.25 \
  --motion_blur_prob 0.10 \
  --scintillation_prob 0.10 \
  "${common_args[@]}"

generate_one "testing_hard" \
  --num_images "${HARD_COUNT}" \
  --output_dir "${HARD_DIR}" \
  --seed 202 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.50 \
  --occ_max 0.85 \
  --soft_focus_strength 0.55 \
  --blur_min 1 --blur_max 5 \
  --noise_std_min 4 --noise_std_max 20 \
  --brightness_min -45 --brightness_max 45 \
  --contrast_min 0.55 --contrast_max 1.45 \
  --motion_blur_prob 0.30 \
  --scintillation_prob 0.25 \
  "${common_args[@]}"

generate_one "testing_extreme" \
  --num_images "${EXTREME_COUNT}" \
  --output_dir "${EXTREME_DIR}" \
  --seed 203 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.85 \
  --occ_max 1.00 \
  --occ_distribution beta_high \
  --soft_focus_strength 0.85 \
  --blur_min 2 --blur_max 7 \
  --noise_std_min 8 --noise_std_max 32 \
  --brightness_min -65 --brightness_max 60 \
  --contrast_min 0.40 --contrast_max 1.55 \
  --motion_blur_prob 0.55 \
  --scintillation_prob 0.45 \
  "${common_args[@]}"

generate_one "testing_small" \
  --num_images "${SMALL_COUNT}" \
  --output_dir "${SMALL_DIR}" \
  --seed 204 \
  --marker_min 36 \
  --marker_max 120 \
  --occ_min 0.50 \
  --occ_max 0.85 \
  --soft_focus_strength 0.50 \
  "${common_args[@]}"

cat <<EOF
==> basic testing suites ready under ${OUTPUT_ROOT}

Benchmark example:
  uv run python src/benchmark.py --runs_dir outputs/runs --suites_dir ${OUTPUT_ROOT}
EOF
