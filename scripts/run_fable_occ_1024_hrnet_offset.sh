#!/usr/bin/env bash
set -euo pipefail

# HRNet-w18 (full) + the winning sub-pixel decode config from Exp-1
# (fable_occ_1024_offset): offset_head_hidden=64, offset_weight=4.0. This is the
# clean A/B that the previous HRNet run (fable_occ_hrnet_w18) never got -- that
# run used the OLD weak offset config (hidden=0, weight=2.0), so its 6.68px
# center-L2 was capped by the decode head, not the backbone. HRNet already had
# the LOWEST val_loss of every run (0.12010 = best heatmap fidelity), so the
# question here is: does that cleaner heatmap finally translate into the best
# center-L2 once it gets a proper offset head?
#
# Everything except the backbone mirrors fable_occ_1024_offset (same data /
# resolution / schedule / loss) so the comparison is single-variable.
#
#   baseline to beat : fable_occ_1024_offset best.pt center-L2 4.34px
#   prior HRNet      : fable_occ_hrnet_w18  best.pt center-L2 6.68px (weak offset)
#
# Memory note: on 16GB cards (GPU 0/1/4) batch 10 OOM'd even with amp -- with amp
# a per-GPU batch of ~10 costs about the same as the no-amp batch-5 the original
# full-HRNet run sat at. Settled on batch 6 (amp + channels_last); if it still
# OOMs drop to 5. If you later move to bigger cards you can push it back up.
#
# Grad accumulation: per-GPU 6 x 3 GPUs = global batch 18, far below the offset
# baseline's 120. --grad_accum_steps 6 accumulates 6 micro-batches per optimizer
# step -> effective global batch 6*3*6 = 108, restoring the optimizer's batch
# (and gradient-noise scale) near the baseline at no extra memory. Caveat: HRNet
# BatchNorm stats are still computed on the 6-sample micro-batch, so this fixes
# the optimizer batch but not small-batch BN noise.

CUDA_VISIBLE_DEVICES=0,1,4 uv run --extra cu126 torchrun --nproc_per_node=3 \
  src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi \
  --train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ \
  --train_dataset_dir outputs/training_sets/real_world_merged_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --train_dataset_dir outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  --output_dir outputs/runs/fable_occ_1024_hrnet_offset \
  --backbone hrnet_w18 \
  --epochs 80 \
  --batch_size 6 \
  --grad_accum_steps 6 \
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
