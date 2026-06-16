#!/usr/bin/env bash
set -euo pipefail

# Experiment 1: lower center-L2 on the resnet18 1024x640 model by giving the
# sub-pixel offset head real capacity instead of a single 1x1 conv, and weighting
# its loss more heavily. Everything else mirrors outputs/runs/fable_occ_1024 so
# this is a clean A/B (same data / backbone / resolution / schedule).
#
#   baseline fable_occ_1024 : offset_head_hidden=0 (1x1 conv), offset_weight=2.0
#   this run                : offset_head_hidden=64 (3x3+ReLU+1x1), offset_weight=4.0

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra cu126 torchrun --nproc_per_node=4 \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ \
  --train_dataset_dir outputs/training_sets/real_world_merged_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  --output_dir outputs/runs/fable_occ_1024_offset \
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
