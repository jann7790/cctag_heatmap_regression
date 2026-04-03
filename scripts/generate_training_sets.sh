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
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/training_sets/generated_training_sets}"

BASE_COUNT="${BASE_COUNT:-4000}"
HARD_COUNT="${HARD_COUNT:-3000}"
EXTREME_COUNT="${EXTREME_COUNT:-1500}"
SMALL_COUNT="${SMALL_COUNT:-1500}"
HARD_NEG_COUNT="${HARD_NEG_COUNT:-3000}"
OVEREXPOSURE_COUNT="${OVEREXPOSURE_COUNT:-2000}"

BASE_CLEAN_COUNT="${BASE_CLEAN_COUNT:-1200}"
BASE_LOW_OCC_COUNT="${BASE_LOW_OCC_COUNT:-2800}"

if [[ $((BASE_CLEAN_COUNT + BASE_LOW_OCC_COUNT)) -ne ${BASE_COUNT} ]]; then
  echo "Base split mismatch: BASE_CLEAN_COUNT + BASE_LOW_OCC_COUNT must equal BASE_COUNT" >&2
  exit 1
fi

common_args=(
  --output_size 640x400
  --marker_style cctag_source
  --num_rings 3
  --heatmap_stride 8
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

merge_dataset_parts() {
  local merged_dir="$1"
  shift
  local parts=("$@")

  mkdir -p "${merged_dir}/images" "${merged_dir}/heatmaps" "${merged_dir}/labels_yolo" "${merged_dir}/config_parts"

  head -n 1 "${parts[0]}/labels.csv" > "${merged_dir}/labels.csv"

  local idx=0
  local part
  for part in "${parts[@]}"; do
    cp "${part}/config.json" "${merged_dir}/config_parts/$(basename "${part}").json"
    while IFS=, read -r filename rest; do
      local new_name
      new_name="$(printf '%06d' "${idx}")"
      cp "${part}/images/${filename}.png" "${merged_dir}/images/${new_name}.png"
      cp "${part}/heatmaps/${filename}.npy" "${merged_dir}/heatmaps/${new_name}.npy"
      cp "${part}/labels_yolo/${filename}.txt" "${merged_dir}/labels_yolo/${new_name}.txt"
      printf '%s,%s\n' "${new_name}" "${rest}" >> "${merged_dir}/labels.csv"
      idx=$((idx + 1))
    done < <(tail -n +2 "${part}/labels.csv")
  done

  cat > "${merged_dir}/README.txt" <<EOF
Merged dataset assembled from:
$(for part in "${parts[@]}"; do printf '%s\n' "  - $(basename "${part}")"; done)

Base set intent:
  - clean subset: approximately 30%% with zero occlusion
  - low-occlusion subset: remaining samples with 5%%-50%% occlusion

The original generator configs for each subset are stored under:
  config_parts/
EOF
}

mkdir -p "${OUTPUT_ROOT}"

BASE_DIR="${OUTPUT_ROOT}/base_set"
HARD_DIR="${OUTPUT_ROOT}/hard_set"
EXTREME_DIR="${OUTPUT_ROOT}/extreme_set"
SMALL_DIR="${OUTPUT_ROOT}/small_set"
HARD_NEG_DIR="${OUTPUT_ROOT}/hard_negative_set"
OVEREXPOSURE_DIR="${OUTPUT_ROOT}/overexposure_set"

ensure_absent "${BASE_DIR}"
ensure_absent "${HARD_DIR}"
ensure_absent "${EXTREME_DIR}"
ensure_absent "${SMALL_DIR}"
ensure_absent "${HARD_NEG_DIR}"
ensure_absent "${OVEREXPOSURE_DIR}"

TMP_ROOT="$(mktemp -d "${OUTPUT_ROOT}/.tmp_training_sets.XXXXXX")"
trap 'rm -rf "${TMP_ROOT}"' EXIT

BASE_CLEAN_DIR="${TMP_ROOT}/base_clean"
BASE_LOW_OCC_DIR="${TMP_ROOT}/base_low_occ"

generate_one "base_clean" \
  --num_images "${BASE_CLEAN_COUNT}" \
  --output_dir "${BASE_CLEAN_DIR}" \
  --seed 42 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.0 \
  --occ_max 0.0 \
  --soft_focus_strength 0.20 \
  "${common_args[@]}"

generate_one "base_low_occ" \
  --num_images "${BASE_LOW_OCC_COUNT}" \
  --output_dir "${BASE_LOW_OCC_DIR}" \
  --seed 43 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.05 \
  --occ_max 0.50 \
  --soft_focus_strength 0.25 \
  "${common_args[@]}"

echo "==> merging base_set"
merge_dataset_parts "${BASE_DIR}" "${BASE_CLEAN_DIR}" "${BASE_LOW_OCC_DIR}"

generate_one "hard_set" \
  --num_images "${HARD_COUNT}" \
  --output_dir "${HARD_DIR}" \
  --seed 44 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.50 \
  --occ_max 0.85 \
  --soft_focus_strength 0.70 \
  "${common_args[@]}"

generate_one "extreme_set" \
  --num_images "${EXTREME_COUNT}" \
  --output_dir "${EXTREME_DIR}" \
  --seed 45 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.90 \
  --occ_max 1.00 \
  --soft_focus_strength 0.90 \
  "${common_args[@]}"

generate_one "small_set" \
  --num_images "${SMALL_COUNT}" \
  --output_dir "${SMALL_DIR}" \
  --seed 46 \
  --marker_min 36 \
  --marker_max 120 \
  --occ_min 0.50 \
  --occ_max 0.85 \
  --soft_focus_strength 0.50 \
  "${common_args[@]}"

# ---------- NEW: Hard negative set ----------
# No marker present, complex backgrounds with curves/arcs/rings that
# resemble CCTag patterns. High negative ratio forces the model to learn
# what is NOT a CCTag.
generate_one "hard_negative_set" \
  --num_images "${HARD_NEG_COUNT}" \
  --output_dir "${HARD_NEG_DIR}" \
  --seed 47 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.0 \
  --occ_max 0.0 \
  --soft_focus_strength 0.30 \
  --background_complexity complex \
  --negative_ratio 0.70 \
  --output_size 640x400 \
  --marker_style cctag_source \
  --num_rings 3 \
  --heatmap_stride 8 \
  --heatmap_sigma 2.0 \
  --partial_out_prob 0.10 \
  --partial_out_max_ratio 0.25 \
  --occlusion_style aggressive

# ---------- NEW: Overexposure set ----------
# Simulates blown-out, bright scenes that cause false positives in
# real-world tracking. Mix of positive and negative samples under
# strong overexposure.
generate_one "overexposure_set" \
  --num_images "${OVEREXPOSURE_COUNT}" \
  --output_dir "${OVEREXPOSURE_DIR}" \
  --seed 48 \
  --marker_min 66 \
  --marker_max 333 \
  --occ_min 0.0 \
  --occ_max 0.50 \
  --soft_focus_strength 0.40 \
  --overexposure_prob 0.80 \
  --background_complexity complex \
  --negative_ratio 0.50 \
  --output_size 640x400 \
  --marker_style cctag_source \
  --num_rings 3 \
  --heatmap_stride 8 \
  --heatmap_sigma 2.0 \
  --partial_out_prob 0.15 \
  --partial_out_max_ratio 0.25 \
  --occlusion_style aggressive

echo "==> training sets ready under ${OUTPUT_ROOT}"
echo "   - ${BASE_DIR}"
echo "   - ${HARD_DIR}"
echo "   - ${EXTREME_DIR}"
echo "   - ${SMALL_DIR}"
echo "   - ${HARD_NEG_DIR}"
echo "   - ${OVEREXPOSURE_DIR}"
