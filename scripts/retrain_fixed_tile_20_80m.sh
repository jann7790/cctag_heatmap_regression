#!/usr/bin/env bash
set -euo pipefail

# From-scratch CCTag retraining for the fixed 1024x640 tile setup.
# This intentionally does not pass --resume_from, so it does not fine-tune from
# outputs/runs/fable_occ_hrnet_w18/epoch_070.pt or any other CCTag checkpoint.

TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-outputs/training_sets/fixed_tile_20_80m_train}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-outputs/datasets/fixed_tile_real_holdout}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/runs/fixed_tile_20_80m_retrain_${RUN_STAMP}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-5}"
EPOCHS="${EPOCHS:-50}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
OFFSET_HEAD_HIDDEN="${OFFSET_HEAD_HIDDEN:-0}"
OFFSET_WEIGHT="${OFFSET_WEIGHT:-2.0}"
SAVE_EVERY="${SAVE_EVERY:-5}"

if [[ ! -d "${TRAIN_DATASET_DIR}" ]]; then
  echo "Missing TRAIN_DATASET_DIR: ${TRAIN_DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -d "${VAL_DATASET_DIR}" ]]; then
  echo "Missing VAL_DATASET_DIR: ${VAL_DATASET_DIR}" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}/metrics.csv" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "Refusing to append to existing run: ${OUTPUT_DIR}" >&2
  echo "Use a new OUTPUT_DIR or set ALLOW_EXISTING_OUTPUT=1." >&2
  exit 1
fi

echo "Fixed-tile retrain"
echo "  train: ${TRAIN_DATASET_DIR}"
echo "  val:   ${VAL_DATASET_DIR}"
echo "  out:   ${OUTPUT_DIR}"
echo "  gpus:  ${CUDA_VISIBLE_DEVICES} (nproc=${NPROC_PER_NODE})"
echo "  batch: per_gpu=${BATCH_SIZE}, accum=${GRAD_ACCUM_STEPS}"
echo "  note:  no --resume_from checkpoint is used"

export CUDA_VISIBLE_DEVICES

uv run torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir "${TRAIN_DATASET_DIR}" \
  --val_dataset_dir "${VAL_DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --backbone hrnet_w18 \
  --input_width 1024 \
  --input_height 640 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --save_every "${SAVE_EVERY}" \
  --focal_loss \
  --offset_head \
  --offset_head_hidden "${OFFSET_HEAD_HIDDEN}" \
  --offset_weight "${OFFSET_WEIGHT}" \
  --size_head \
  --scale_balanced_sampler \
  --positive_fraction 0.6 \
  --augment \
  --aug_brightness 0.2 \
  --aug_contrast 0.2 \
  --aug_noise_std 6.0 \
  --amp
