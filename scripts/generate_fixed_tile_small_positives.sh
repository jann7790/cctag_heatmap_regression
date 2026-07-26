#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
GENERATOR="${GENERATOR:-src/generate_cctag_dataset.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training_sets/fixed_tile_20_80m_supplement_no_occ_v2}"
BIN_48_80_COUNT="${BIN_48_80_COUNT:-4500}"
BIN_80_128_COUNT="${BIN_80_128_COUNT:-4500}"
BIN_128_192_COUNT="${BIN_128_192_COUNT:-3000}"
BIN_192_320_COUNT="${BIN_192_320_COUNT:-3000}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-200}"
SHARDS="${SHARDS:-4}"
CLEAN_PERCENT="${CLEAN_PERCENT:-65}"
DEGRADED_PERCENT="${DEGRADED_PERCENT:-25}"
BOUNDARY_PERCENT="${BOUNDARY_PERCENT:-10}"

if ((SHARDS <= 0)); then
  echo "SHARDS must be positive" >&2
  exit 1
fi

if ((CLEAN_PERCENT < 0 || DEGRADED_PERCENT < 0 || BOUNDARY_PERCENT < 0 || CLEAN_PERCENT + DEGRADED_PERCENT + BOUNDARY_PERCENT != 100)); then
  echo "CLEAN_PERCENT, DEGRADED_PERCENT, and BOUNDARY_PERCENT must be non-negative and sum to 100" >&2
  exit 1
fi

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing path: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

common=(
  --output_size 1024x640
  --marker_style cctag_source
  --num_rings 3
  --heatmap_stride 4
  --heatmap_sigma 3.0
  --actual_diameter_max_attempts 150
  --low_light_prob 0.50
  --low_light_white_min 35 --low_light_white_max 100
  --low_light_black_min 4 --low_light_black_max 20
  --vignette_prob 0.40
  --vignette_strength_min 0.30 --vignette_strength_max 0.75
)

generate_part() {
  local output="$1"
  local count="$2"
  local seed="$3"
  local actual_min="$4"
  local actual_max="$5"
  local marker_min="$6"
  local marker_max="$7"
  local mode="$8"

  local mode_args=()
  case "${mode}" in
    clean)
      mode_args=(
        --negative_ratio 0
        --required_target_clamped 0
        --visible_marker_ratio_min 0.999
        --occ_min 0.0 --occ_max 0.0
        --partial_out_prob 0.0
        --blur_min 0 --blur_max 1
        --noise_std_min 0 --noise_std_max 6
        --brightness_min -12 --brightness_max 12
        --contrast_min 0.85 --contrast_max 1.15
        --motion_blur_prob 0.03
        --scintillation_prob 0.03
      )
      ;;
    degraded)
      mode_args=(
        --negative_ratio 0
        --required_target_clamped 0
        --visible_marker_ratio_min 0.999
        --occ_min 0.0 --occ_max 0.0
        --partial_out_prob 0.0
        --blur_min 0 --blur_max 3
        --noise_std_min 2 --noise_std_max 16
        --brightness_min -35 --brightness_max 35
        --contrast_min 0.60 --contrast_max 1.35
        --motion_blur_prob 0.20
        --scintillation_prob 0.30
        --overexposure_prob 0.25
      )
      ;;
    boundary)
      mode_args=(
        --empty_negative_ratio 0 --boundary_target_ratio 1
        --occ_min 0.0 --occ_max 0.0
        --required_target_clamped 1
        --visible_marker_ratio_min 0.25 --visible_marker_ratio_max 0.998999
        --partial_out_max_ratio 0.40
        --blur_min 0 --blur_max 2
        --noise_std_min 0 --noise_std_max 12
        --brightness_min -25 --brightness_max 25
        --contrast_min 0.70 --contrast_max 1.25
        --motion_blur_prob 0.10
        --scintillation_prob 0.15
      )
      ;;
    negative)
      mode_args=(
        --negative_ratio 1
        --occ_min 0.0 --occ_max 0.0
        --partial_out_prob 0.0
        --background_complexity complex
        --blur_min 0 --blur_max 2
        --noise_std_min 0 --noise_std_max 14
        --brightness_min -30 --brightness_max 30
        --contrast_min 0.65 --contrast_max 1.30
        --motion_blur_prob 0.10
        --scintillation_prob 0.10
      )
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      exit 1
      ;;
  esac

  "${PYTHON_BIN}" "${GENERATOR}" \
    --num_images "${count}" \
    --output_dir "${output}" \
    --seed "${seed}" \
    --marker_min "${marker_min}" \
    --marker_max "${marker_max}" \
    --actual_diameter_min "${actual_min}" \
    --actual_diameter_max "${actual_max}" \
    "${common[@]}" \
    "${mode_args[@]}"
}

pids=()
job_names=()

launch_part() {
  local output="$1"
  generate_part "$@" &
  pids+=("$!")
  job_names+=("${output}")
}

launch_shards() {
  local name="$1"
  local total="$2"
  local seed_base="$3"
  local actual_min="$4"
  local actual_max="$5"
  local marker_min="$6"
  local marker_max="$7"
  local mode="$8"
  local base_count=$((total / SHARDS))
  local remainder=$((total % SHARDS))

  for ((shard = 0; shard < SHARDS; shard++)); do
    local count="${base_count}"
    if ((shard < remainder)); then
      count=$((count + 1))
    fi
    if ((count == 0)); then
      continue
    fi
    launch_part \
      "${OUTPUT_ROOT}/${name}_${mode}_$(printf '%02d' "${shard}")" \
      "${count}" "$((seed_base + shard))" \
      "${actual_min}" "${actual_max}" "${marker_min}" "${marker_max}" "${mode}"
  done
}

generate_bin() {
  local name="$1"
  local total="$2"
  local actual_min="$3"
  local actual_max="$4"
  local marker_min="$5"
  local marker_max="$6"
  local seed_base="$7"
  local clean_count=$((total * CLEAN_PERCENT / 100))
  local degraded_count=$((total * DEGRADED_PERCENT / 100))
  local boundary_count=$((total - clean_count - degraded_count))

  launch_shards "${name}" "${clean_count}" "$((seed_base + 0))" \
    "${actual_min}" "${actual_max}" "${marker_min}" "${marker_max}" clean
  launch_shards "${name}" "${degraded_count}" "$((seed_base + 100))" \
    "${actual_min}" "${actual_max}" "${marker_min}" "${marker_max}" degraded
  launch_shards "${name}" "${boundary_count}" "$((seed_base + 200))" \
    "${actual_min}" "${actual_max}" "${marker_min}" "${marker_max}" boundary
}

generate_bin "48_80" "${BIN_48_80_COUNT}" 48 79.999 20 80 100
generate_bin "80_128" "${BIN_80_128_COUNT}" 80 127.999 35 128 200
generate_bin "128_192" "${BIN_128_192_COUNT}" 128 191.999 55 192 300
generate_bin "192_320" "${BIN_192_320_COUNT}" 192 319.999 80 320 400
launch_shards "hard_negative" "${NEGATIVE_COUNT}" 400 48 320 24 80 negative

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "Generation failed: ${job_names[index]}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "Generated fixed-tile small-positive supplement at ${OUTPUT_ROOT}"
echo "Pass all child directories to scripts/build_fixed_tile_dataset.py with existing train-only sources."
