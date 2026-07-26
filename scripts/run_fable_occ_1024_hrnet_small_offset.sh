#!/usr/bin/env bash
set -euo pipefail

# HRNet-w18-small-v2 (lighter HRNet: fewer modules/blocks) + the winning offset
# config (hidden=64, weight=4.0). Paired with run_fable_occ_1024_hrnet_offset.sh
# to answer the cost/benefit question directly:
#
#   - full hrnet_w18 was ~4x train / ~10x infer vs resnet18.
#   - small_v2 cuts that cost, but ALSO trims the multi-branch capacity that
#     gave HRNet its best-in-class heatmap fidelity (val_loss).
#
# Compare this run's val_loss + center-L2 against:
#   - fable_occ_1024_hrnet_offset  (full HRNet, same offset config)  -> how much
#     accuracy the shrink costs.
#   - fable_occ_1024_offset 4.34px (resnet18, same offset config)    -> whether a
#     small HRNet is even worth it over plain resnet18 at similar cost.
#
# Single-variable vs the full-HRNet run: ONLY --backbone changes. small_v2 trims
# channels/blocks but keeps HRNet's high-res branch activations (the real memory
# cost at 1024x640), so on 16GB cards batch 20 OOM'd -- settled on batch 10 (vs
# the full run's 6). Drop further if it still OOMs.
#
# Grad accumulation: 10 x 3 GPUs = global 30; --grad_accum_steps 4 -> effective
# global batch 10*3*4 = 120, matching the offset baseline. Same BN caveat as the
# full-HRNet script (stats still computed on the 10-sample micro-batch).

CUDA_VISIBLE_DEVICES=0,1,4 uv run --extra cu126 torchrun --nproc_per_node=3 \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ \
  --train_dataset_dir outputs/training_sets/real_world_merged_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  --output_dir outputs/runs/fable_occ_1024_hrnet_small_offset \
  --backbone hrnet_w18_small_v2 \
  --epochs 80 \
  --batch_size 10 \
  --grad_accum_steps 4 \
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
