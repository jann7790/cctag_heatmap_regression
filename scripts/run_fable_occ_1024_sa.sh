#!/usr/bin/env bash
set -euo pipefail

# Experiment 2 (size-adaptive sigma + sharper-peak loss):
# Same backbone/resolution/schedule as Exp-1 (fable_occ_1024_offset), but with:
#   - Size-adaptive sigma heatmap targets (_sa datasets)
#   - A1: dice_weight=0.3 (less dice → sharper per-pixel peaks)
#   - A2: concentration_weight=0.05 (pull predicted mass to GT center)
#
# Ablation guide (toggle individually to isolate effects):
#   - A1 only : --dice_weight 0.3 --concentration_weight 0.0
#   - A2 only : --dice_weight 0.5 --concentration_weight 0.05
#   - Neither : --dice_weight 0.5 --concentration_weight 0.0  (Exp-1 loss on _sa data)
#   - Both    : --dice_weight 0.3 --concentration_weight 0.05 (default below)

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra cu126 torchrun --nproc_per_node=4 \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset_sa \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_sa \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ_sa \
  --train_dataset_dir outputs/training_sets/real_world_merged_1024x640_sa \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  --output_dir outputs/runs/fable_occ_1024_sa \
  --backbone resnet18 \
  --epochs 80 \
  --batch_size 30 \
  --lr 0.0014 \
  --weight_decay 0.0001 \
  --train_ratio 0.9 \
  --seed 42 \
  --input_width 1024 \
  --input_height 640 \
  --offset_head \
  --offset_weight 4.0 \
  --offset_head_hidden 64 \
  --size_head \
  --size_weight 1.0 \
  --focal_loss \
  --amp \
  --channels_last \
  --dice_weight 0.3 \
  --concentration_weight 0.05 \
  --concentration_window 7
