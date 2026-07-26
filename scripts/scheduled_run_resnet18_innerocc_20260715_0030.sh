#!/usr/bin/env bash
# One-shot launcher for 2026-07-15 00:30 Asia/Taipei.
set -euo pipefail

cd /home/r13922171/dataset
mkdir -p outputs/logs

exec env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NPROC_PER_NODE=4 \
  BATCH_SIZE=15 \
  GRAD_ACCUM_STEPS=2 \
  bash scripts/run_fixed_tile_20_80m_resnet18_innerocc.sh \
  >> outputs/logs/resnet18_innerocc_20260715_0030.log 2>&1
