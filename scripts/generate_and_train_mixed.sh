#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
GENERATOR_SCRIPT="${GENERATOR_SCRIPT:-./scripts/generate_training_sets.sh}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./src/train_cctag_heatmap_ddp.py}"

TRAINING_ROOT="${TRAINING_ROOT:-./outputs/training_sets/generated_training_sets}"
MIXED_DATASET_DIR="${MIXED_DATASET_DIR:-${TRAINING_ROOT}/mixed_train_dataset}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-./outputs/runs/experiment_mixed}"

merge_dataset_parts() {
  local merged_dir="$1"
  shift
  local parts=("$@")

  if [[ ${#parts[@]} -eq 0 ]]; then
    echo "No dataset parts were provided for merge." >&2
    exit 1
  fi

  if [[ -e "${merged_dir}" ]]; then
    echo "Refusing to overwrite existing path: ${merged_dir}" >&2
    exit 1
  fi

  mkdir -p "${merged_dir}/images" "${merged_dir}/heatmaps" "${merged_dir}/labels_yolo" "${merged_dir}/config_parts"
  head -n 1 "${parts[0]}/labels.csv" > "${merged_dir}/labels.csv"

  local idx=0
  local part
  for part in "${parts[@]}"; do
    if [[ -f "${part}/config.json" ]]; then
      cp "${part}/config.json" "${merged_dir}/config_parts/$(basename "${part}").json"
    elif [[ -d "${part}/config_parts" ]]; then
      local config_part
      shopt -s nullglob
      for config_part in "${part}"/config_parts/*.json; do
        cp "${config_part}" "${merged_dir}/config_parts/$(basename "${part}")_$(basename "${config_part}")"
      done
      shopt -u nullglob
    else
      echo "Missing config metadata for dataset part: ${part}" >&2
      exit 1
    fi

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

  {
    printf 'Mixed training dataset merged from:\n'
    for part in "${parts[@]}"; do
      printf '  - %s\n' "$(basename "${part}")"
    done
    printf '\nTotal samples:\n  %s\n\n' "${idx}"
    printf 'Original per-set generator configs are stored under:\n  config_parts/\n'
  } > "${merged_dir}/README.txt"
}

echo "==> generating training sets"
OUTPUT_ROOT="${TRAINING_ROOT}" "${GENERATOR_SCRIPT}"

BASE_DIR="${TRAINING_ROOT}/base_set"
HARD_DIR="${TRAINING_ROOT}/hard_set"
EXTREME_DIR="${TRAINING_ROOT}/extreme_set"
SMALL_DIR="${TRAINING_ROOT}/small_set"
HARD_NEG_DIR="${TRAINING_ROOT}/hard_negative_set"
OVEREXPOSURE_DIR="${TRAINING_ROOT}/overexposure_set"

for required_dir in "${BASE_DIR}" "${HARD_DIR}" "${EXTREME_DIR}" "${SMALL_DIR}" "${HARD_NEG_DIR}" "${OVEREXPOSURE_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Missing expected dataset directory: ${required_dir}" >&2
    exit 1
  fi
done

echo "==> merging datasets into ${MIXED_DATASET_DIR}"
merge_dataset_parts "${MIXED_DATASET_DIR}" "${BASE_DIR}" "${HARD_DIR}" "${EXTREME_DIR}" "${SMALL_DIR}" "${HARD_NEG_DIR}" "${OVEREXPOSURE_DIR}"

NPROC="${NPROC:-4}"

echo "==> starting training (DDP, nproc=${NPROC})"
"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC}" "${TRAIN_SCRIPT}" \
  --dataset_dir "${MIXED_DATASET_DIR}" \
  --output_dir "${RUN_OUTPUT_DIR}" \
  --epochs 80 \
  --batch_size 18 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --train_ratio 0.9 \
  --num_workers 8 \
  --seed 42 \
  --input_width 640 \
  --input_height 400 \
  --save_every 10 \
  --focal_loss
