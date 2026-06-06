#!/usr/bin/env bash
# v2 pipeline: generate tuned detection data -> build YOLO layout -> retrain.
# Tuned vs original: near-subset marker_max 300->450 (diam up to 900px),
# low_light_white 45-90 -> 25-70 (darker, matches real FLIR mean ~20).
set -euo pipefail
cd /home/r13922171/dataset

ROOT=/home/r13922171/dataset
PY="${ROOT}/.venv/bin/python"
GEN_ROOT="${ROOT}/outputs/training_sets/detection_sets_v2"
DATA_DIR="${ROOT}/outputs/datasets/yolo_detection_v2"

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] === STAGE 1/3: generate v2 detection sets ==="
PYTHON_BIN="${PY}" bash scripts/generate_detection_set_v2.sh

echo "[$(ts)] === STAGE 2/3: build ultralytics layout ==="
"${PY}" src/prepare_yolo_dataset.py \
  --dataset_dir "${GEN_ROOT}/det_positive_wide" \
  --dataset_dir "${GEN_ROOT}/det_hard" \
  --dataset_dir "${GEN_ROOT}/det_hard_negative" \
  --dataset_dir "${GEN_ROOT}/det_overexposure" \
  --dataset_dir "${GEN_ROOT}/det_far_small" \
  --output_dir "${DATA_DIR}"

echo "[$(ts)] === STAGE 3/3: train YOLO (30 epochs) ==="
"${PY}" src/train_yolo_detection.py \
  --data "${DATA_DIR}/data.yaml" \
  --model yolo11n.pt \
  --epochs 30 --imgsz 1024 --batch 80 \
  --device 0,2 --workers 16 --cache ram \
  --project "${ROOT}/outputs/runs_yolo" \
  --name cctag_det_n_v2

echo "[$(ts)] === DONE. weights: outputs/runs_yolo/cctag_det_n_v2/weights/best.pt ==="
