#!/usr/bin/env bash
set -euo pipefail

# Required inputs are kept explicit so a run cannot silently fall back to an
# internally mixed frame split or start from ImageNet weights.
: "${TRAIN_DATASET_DIR:?set TRAIN_DATASET_DIR to the session-split training dataset}"
: "${VAL_DATASET_DIR:?set VAL_DATASET_DIR to the session-split validation dataset}"
: "${CHECKPOINT:?set CHECKPOINT to the existing HRNet-W18 checkpoint}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/runs/fixed_tile_20_80m}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-5}"
EPOCHS="${EPOCHS:-25}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-6}"
OFFSET_HEAD_HIDDEN="${OFFSET_HEAD_HIDDEN:-0}"
OFFSET_WEIGHT="${OFFSET_WEIGHT:-2.0}"

uv run torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir "${TRAIN_DATASET_DIR}" \
  --val_dataset_dir "${VAL_DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from "${CHECKPOINT}" \
  --backbone hrnet_w18 \
  --input_width 1024 \
  --input_height 640 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --lr 1e-4 \
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
