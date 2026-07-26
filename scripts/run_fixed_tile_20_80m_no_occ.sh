#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the existing HRNet-W18 detector on the no-occlusion fixed-tile set.
# Model weights come from CHECKPOINT; optimizer state is deliberately new.
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-outputs/training_sets/fixed_tile_20_80m_train_no_occ}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-outputs/datasets/fixed_tile_real_holdout}"
CHECKPOINT="${CHECKPOINT:-outputs/runs/fable_occ_hrnet_w18/epoch_070.pt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/runs/fixed_tile_20_80m/no_occlusion_finetune_${RUN_STAMP}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-5}"
EPOCHS="${EPOCHS:-25}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"
OFFSET_HEAD_HIDDEN="${OFFSET_HEAD_HIDDEN:-0}"
OFFSET_WEIGHT="${OFFSET_WEIGHT:-2.0}"

for path in "${TRAIN_DATASET_DIR}" "${VAL_DATASET_DIR}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing dataset directory: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi

uv run python scripts/verify_fixed_tile_dataset.py "${TRAIN_DATASET_DIR}"
uv run python scripts/verify_fixed_tile_dataset.py --holdout "${VAL_DATASET_DIR}"

echo "No-occlusion fixed-tile fine-tune"
echo "  train: ${TRAIN_DATASET_DIR}"
echo "  val:   ${VAL_DATASET_DIR}"
echo "  ckpt:  ${CHECKPOINT}"
echo "  out:   ${OUTPUT_DIR}"
echo "  batch: per_gpu=${BATCH_SIZE}, accum=${GRAD_ACCUM_STEPS}"

export CUDA_VISIBLE_DEVICES

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
