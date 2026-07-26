#!/usr/bin/env bash
set -euo pipefail

# Fine-tune the ResNet-18 offset model from fable_occ_1024_offset on fixed-tile
# V4 (strict lower-corner proxies + inner-ring center occlusion). --resume_from
# loads weights only; the optimizer, scheduler, and epoch counter start fresh.
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-/mnt/tmp1/r13922171/fixed_tile_20_80m_train_innerocc_corner_v4}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-/mnt/tmp1/r13922171/fixed_tile_real_holdout_v2}"
CHECKPOINT="${CHECKPOINT:-outputs/runs/fable_occ_1024_offset/best.pt}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/runs/fixed_tile_20_80m_resnet18/innerocc_corner_v4_${RUN_STAMP}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
# 15 GPUs samples × 4 processes × 2 accumulation steps = effective batch 120.
# The source ResNet-18 run used 30 samples/process; 15 is a safer default for
# 16 GB cards while still substantially larger than the HRNet setting (6).
BATCH_SIZE="${BATCH_SIZE:-15}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EPOCHS="${EPOCHS:-25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"

for path in "${TRAIN_DATASET_DIR}" "${VAL_DATASET_DIR}"; do
  [[ -d "${path}" ]] || { echo "Missing dataset directory: ${path}" >&2; exit 1; }
done
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite: ${OUTPUT_DIR}" >&2; exit 1; }

uv run python scripts/verify_fixed_tile_dataset.py "${TRAIN_DATASET_DIR}"
uv run python scripts/verify_fixed_tile_dataset.py --holdout "${VAL_DATASET_DIR}"

export CUDA_VISIBLE_DEVICES
uv run torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir "${TRAIN_DATASET_DIR}" \
  --val_dataset_dir "${VAL_DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume_from "${CHECKPOINT}" \
  --backbone resnet18 \
  --input_width 1024 --input_height 640 \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" \
  --save_every "${SAVE_EVERY}" \
  --focal_loss \
  --offset_head --offset_head_hidden 64 --offset_weight 4.0 \
  --size_head --size_weight 1.0 \
  --scale_balanced_sampler --positive_fraction 0.6 \
  --augment --aug_brightness 0.2 --aug_contrast 0.2 --aug_noise_std 6.0 \
  --amp
