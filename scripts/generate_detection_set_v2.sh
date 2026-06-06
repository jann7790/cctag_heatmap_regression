#!/usr/bin/env bash
# Generate a DETECTION-tuned dataset for the YOLO-nano acquisition model.
#
# Design intent (see plan: YOLO-nano detection + two-stage localization):
#   - WIDE marker size range  -> wide "firing band", fewer acquisition scales.
#   - HEAVY negatives / hard negatives / overexposure -> precision-leaning detector
#     (false positives waste an ROI crop; we bias against them).
#   - HIGH partial-out         -> markers at frame edges during camera search.
#
# Output is already YOLO-ready: each subset has images/*.png + labels_yolo/*.txt
# (positives carry "class cx cy w h"; negatives are EMPTY files = background).
# Feed all subset dirs into src/prepare_yolo_dataset.py to build the ultralytics
# layout + data.yaml. (Generated heatmaps/ are unused by YOLO and can be ignored.)
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
GENERATOR="${GENERATOR:-./src/generate_cctag_dataset.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/training_sets/detection_sets_v2}"

POS_COUNT="${POS_COUNT:-5000}"
HARD_COUNT="${HARD_COUNT:-3000}"
NEG_COUNT="${NEG_COUNT:-3000}"
OVEREXPOSURE_COUNT="${OVEREXPOSURE_COUNT:-2000}"
FAR_COUNT="${FAR_COUNT:-4000}"

# Common generation knobs shared by every subset. --output_size is per-subset, all
# at the 1.9 aspect of a real 2x2 tile (2048x1080) so letterbox padding matches
# deployment: the four near/medium subsets render at 1024x540 (near markers do not
# need extra pixels), det_far_small at 2048x1080 (small markers need native px).
# Marker size bands are continuous: far_small 1.5-8% of width, near subsets ~5-60%.
# YOLO resizes every input to imgsz (1024), so what matters is marker-as-%-of-width,
# which these match to the deployment tile.
common_args=(
  --marker_style cctag_source
  --num_rings 3
  --no_heatmap
  --empty_negative_ratio 0.15
  --boundary_target_ratio 0.10
  --occlusion_style aggressive
)

# New-lens (40m) domain: heavily underexposed, sharp, strong corner vignetting.
# Same knobs as scripts/generate_training_sets.sh so the YOLO detector trains on
# the same dark/sharp/vignetted look the deployed camera produces (replaces the
# old plano-convex bright + soft-focus domain). Sizes stay at the deploy-tile
# resolutions (1024x540 / 2048x1080); only the photometric domain changes.
lowlight_args=(
  --low_light_prob 0.85
  --low_light_white_min 25 --low_light_white_max 70
  --low_light_black_min 4 --low_light_black_max 18
  --vignette_prob 0.6
  --vignette_strength_min 0.4 --vignette_strength_max 0.9
)

ensure_absent() {
  local target="$1"
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite existing path: ${target}" >&2
    exit 1
  fi
}

generate_one() {
  local name="$1"; shift
  echo "==> generating ${name}"
  "${PYTHON_BIN}" "${GENERATOR}" "$@"
}

mkdir -p "${OUTPUT_ROOT}"
POS_DIR="${OUTPUT_ROOT}/det_positive_wide"
HARD_DIR="${OUTPUT_ROOT}/det_hard"
NEG_DIR="${OUTPUT_ROOT}/det_hard_negative"
OVEREXPOSURE_DIR="${OUTPUT_ROOT}/det_overexposure"
FAR_DIR="${OUTPUT_ROOT}/det_far_small"

ensure_absent "${POS_DIR}"
ensure_absent "${HARD_DIR}"
ensure_absent "${NEG_DIR}"
ensure_absent "${OVEREXPOSURE_DIR}"
ensure_absent "${FAR_DIR}"

# ---------- Positives across a WIDE size range (the firing-band widener) ----------
generate_one "det_positive_wide" \
  --num_images "${POS_COUNT}" \
  --output_dir "${POS_DIR}" \
  --seed 142 \
  --output_size 1024x540 \
  --marker_min 24 \
  --marker_max 450 \
  --occ_min 0.0 \
  --occ_max 0.40 \
  --soft_focus_strength 0.0 \
  --blur_min 0 --blur_max 1 \
  --noise_std_min 0 --noise_std_max 12 \
  --brightness_min -25 --brightness_max 25 \
  --contrast_min 0.75 --contrast_max 1.25 \
  --motion_blur_prob 0.08 \
  --scintillation_prob 0.05 \
  --partial_out_prob 0.35 \
  --partial_out_max_ratio 0.35 \
  "${common_args[@]}" \
  "${lowlight_args[@]}"

# ---------- Hard positives (heavy occlusion + degradation), still wide size ----------
generate_one "det_hard" \
  --num_images "${HARD_COUNT}" \
  --output_dir "${HARD_DIR}" \
  --seed 143 \
  --output_size 1024x540 \
  --marker_min 24 \
  --marker_max 450 \
  --occ_min 0.40 \
  --occ_max 0.85 \
  --soft_focus_strength 0.0 \
  --blur_min 0 --blur_max 2 \
  --noise_std_min 4 --noise_std_max 20 \
  --brightness_min -45 --brightness_max 45 \
  --contrast_min 0.55 --contrast_max 1.45 \
  --motion_blur_prob 0.20 \
  --scintillation_prob 0.10 \
  --partial_out_prob 0.35 \
  --partial_out_max_ratio 0.35 \
  "${common_args[@]}" \
  "${lowlight_args[@]}"

# ---------- Hard negatives: complex backgrounds (curves/arcs/rings), no marker ----------
generate_one "det_hard_negative" \
  --num_images "${NEG_COUNT}" \
  --output_dir "${NEG_DIR}" \
  --seed 144 \
  --output_size 1024x540 \
  --marker_min 24 \
  --marker_max 450 \
  --occ_min 0.0 \
  --occ_max 0.0 \
  --soft_focus_strength 0.0 \
  --blur_min 0 --blur_max 1 \
  --background_complexity complex \
  --negative_ratio 0.80 \
  --partial_out_prob 0.10 \
  --partial_out_max_ratio 0.25 \
  "${common_args[@]}" \
  "${lowlight_args[@]}"

# ---------- Overexposure: blown-out scenes, mix of positives and negatives ----------
# New lens is dark, not blown out, so this set is kept as an FP guard against the
# bright off-center hot-spot occasionally seen in real captures: low low-light prob
# (mostly bright/overexposed), strong vignetting so the bright region stays local.
generate_one "det_overexposure" \
  --num_images "${OVEREXPOSURE_COUNT}" \
  --output_dir "${OVEREXPOSURE_DIR}" \
  --seed 145 \
  --output_size 1024x540 \
  --marker_min 24 \
  --marker_max 450 \
  --occ_min 0.0 \
  --occ_max 0.40 \
  --soft_focus_strength 0.0 \
  --blur_min 0 --blur_max 1 \
  --overexposure_prob 0.80 \
  --low_light_prob 0.20 \
  --low_light_white_min 25 --low_light_white_max 70 \
  --low_light_black_min 4 --low_light_black_max 18 \
  --vignette_prob 0.7 \
  --vignette_strength_min 0.4 --vignette_strength_max 0.9 \
  --background_complexity complex \
  --negative_ratio 0.50 \
  --partial_out_prob 0.15 \
  --partial_out_max_ratio 0.25 \
  "${common_args[@]}"

# ---------- FAR / small markers, rendered at the real 2x2 tile size ----------
# Canvas = 2048x1080 = a single 2x2 tile of a 4096x2160 frame. marker_min/max are
# the marker OUTER-RING RADIUS in px (final diameter = value*2), so 16..82 ->
# 32..164px diameter (1.5-8% of width) = the far-distance band that the smaller
# 1024x540 near subsets cannot render crisply. Deploy must match: `--tile 2x2 --imgsz 1024`.
generate_one "det_far_small" \
  --num_images "${FAR_COUNT}" \
  --output_dir "${FAR_DIR}" \
  --seed 146 \
  --output_size 2048x1080 \
  --marker_min 16 \
  --marker_max 82 \
  --occ_min 0.0 \
  --occ_max 0.40 \
  --soft_focus_strength 0.0 \
  --blur_min 0 --blur_max 1 \
  --noise_std_min 2 --noise_std_max 16 \
  --brightness_min -35 --brightness_max 35 \
  --contrast_min 0.65 --contrast_max 1.35 \
  --motion_blur_prob 0.10 \
  --scintillation_prob 0.10 \
  --partial_out_prob 0.25 \
  --partial_out_max_ratio 0.30 \
  --negative_ratio 0.25 \
  --background_complexity complex \
  "${common_args[@]}" \
  "${lowlight_args[@]}"

echo "==> detection subsets ready under ${OUTPUT_ROOT}"
echo "   - ${POS_DIR}"
echo "   - ${HARD_DIR}"
echo "   - ${NEG_DIR}"
echo "   - ${OVEREXPOSURE_DIR}"
echo "   - ${FAR_DIR}   (2048x1080 far/small markers)"
echo
echo "Next: build the ultralytics layout + data.yaml, e.g."
echo "  ${PYTHON_BIN} src/prepare_yolo_dataset.py \\"
echo "    --dataset_dir ${POS_DIR} --dataset_dir ${HARD_DIR} \\"
echo "    --dataset_dir ${NEG_DIR} --dataset_dir ${OVEREXPOSURE_DIR} \\"
echo "    --dataset_dir ${FAR_DIR} \\"
echo "    --output_dir ./outputs/datasets/yolo_detection"
