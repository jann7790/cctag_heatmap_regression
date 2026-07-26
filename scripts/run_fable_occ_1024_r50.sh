#!/usr/bin/env bash
set -euo pipefail

# resnet50 capacity experiment: single-variable change vs fable_occ_1024_deepdec.
# The ONLY difference is --backbone resnet50 (4x wider bottleneck, ~2-3x params)
# on top of the same --decoder_blocks 2 + --offset_head_hidden 64 stack. Same
# datasets / loss as deepdec so any center-L2 change is attributable to the
# bigger encoder alone.
#
# Baselines to beat: fable_occ_1024_offset 3.79px, fable_occ_1024_deepdec 4.27px.
# NOTE: resnet50 at 1024x640 needs much more VRAM -- batch dropped 30 -> 16.
# If it still OOMs, lower --batch_size further (12/8) or drop --channels_last.

CUDA_VISIBLE_DEVICES=0,1,4 uv run --extra cu126 torchrun --nproc_per_node=3 \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ \
  --train_dataset_dir outputs/training_sets/real_world_merged_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  --output_dir outputs/runs/fable_occ_1024_resnet50 \
  --backbone resnet50 \
  --epochs 80 \
  --batch_size 10 \
  --lr 0.0007 \
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
  --decoder_blocks 2
