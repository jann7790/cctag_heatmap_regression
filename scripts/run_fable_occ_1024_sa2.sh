#!/usr/bin/env bash
set -euo pipefail

# Adaptive-sigma experiment (SAFE single-variable): ONLY change vs Exp-1 is
# the dataset heatmaps — size-adaptive Gaussian sigma (min=1.5, max=3.0, k=0.15)
# instead of fixed sigma 2-3. All other knobs = exp1 exactly.
#
# Baseline to beat: fable_occ_1024_offset best.pt center-L2 4.34px (interior ~4.0).

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra cu126 torchrun --nproc_per_node=4 \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset_sa2 \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_sa2 \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ_sa2 \
  --train_dataset_dir outputs/training_sets/real_world_merged_1024x640_sa2 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  --output_dir outputs/runs/fable_occ_1024_sa2 \
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
  --channels_last
