# Commands

## INFERENCE (tiled, 4K → 1024 native tiles)

兩個 tile 腳本都吃**單張** `--source`(資料夾請用 bash for-loop)。
4096x2160 會被切成 1024 原生視窗(overlap 30% + NMS/去重),保留小 marker 的 px。

### YOLO 偵測(tiled)  →  bounding box
```bash
CUDA_VISIBLE_DEVICES=1 uv run python src/tile_detect.py \
  --model outputs/runs_yolo/cctag_det_n_v2/weights/best.pt \
  --source 40m_example.png \
  --output outputs/runs_yolo/flir_v2_c07 \
  --imgsz 1024 --tile 1024 --overlap 0.3 --conf 0.7 --iou 0.5
# conf 0.7 = v2 的甜蜜點(清暗圖假框又留真框);--save_tiles 可存每塊
```

### CNN heatmap(tiled)  →  次像素圓心 (+ size head 圈出大小)
```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/tile_heatmap.py \
  --checkpoint outputs/runs/heatmap_1024/best.pt \
  --source 40m_example.png \
  --output outputs/inference/hm1024_upgraded \
  --threshold 0.5 --min_peak_sharpness 3.0 --max_size_frac 1.2
# offset head 出次像素圓心;min_peak_sharpness 砍鈍峰 FP;max_size_frac 砍超大尺寸 FP
# --no_offset 可關 offset head(退回 argmax+拋物線)
# tile 尺寸自動讀 checkpoint config 的 input_width/height (= 1024x640)
```

### 整個資料夾批次(範例:跑 21 張 FLIR)
```bash
D="drive-download-20260605T084325Z-3-001"; CK="outputs/runs/heatmap_1024/best.pt"
for f in "$D"/*.png; do
  CUDA_VISIBLE_DEVICES=0 uv run python src/tile_heatmap.py \
    --checkpoint "$CK" --source "$f" --output outputs/inference/hm1024_upgraded \
    --threshold 0.5 --min_peak_sharpness 3.0 --max_size_frac 1.2
done
```

### 目前模型權重
| 模型 | 權重 | 備註 |
|------|------|------|
| YOLO v2 | `outputs/runs_yolo/cctag_det_n_v2/weights/best.pt` | marker_max 450 + 更暗 low-light,conf 0.7 |
| CNN 1024 | `outputs/runs/heatmap_1024/best.pt` | 1024x640, offset+size head, focal loss |


## TRAINING (this session)

### YOLO v2 偵測(30 epoch)
```bash
uv run python src/train_yolo_detection.py \
  --data outputs/datasets/yolo_detection_v2/data.yaml \
  --model yolo11n.pt --epochs 30 --imgsz 1024 --batch 80 \
  --device 0,2 --workers 16 --cache ram \
  --project /home/r13922171/dataset/outputs/runs_yolo --name cctag_det_n_v2
# 資料先用 scripts/generate_detection_set_v2.sh 生成 + src/prepare_yolo_dataset.py 組裝
```

### CNN heatmap 1024 (offset+size head, 4-GPU)
```bash
CUDA_VISIBLE_DEVICES=0,2,3,5 uv run torchrun --nproc_per_node=4 \
  src/train_cctag_heatmap_ddp.py \
  --dataset_dir outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
  --input_width 1024 --input_height 640 \
  --batch_size 20 \
  --offset_head --size_head \
  --focal_loss --focal_gamma 2.0 \
  --output_dir outputs/runs/heatmap_1024
# input_width/height 必須 = 生成的圖尺寸 (1024x640),否則會 resize 失去尺度對齊
# 資料: scripts/generate_training_sets_1024.sh
```


## TRAINING (older reference)

```bash
CUDA_VISIBLE_DEVICES=2,3,4 uv run torchrun --nproc_per_node=3 \
      src/train_cctag_heatmap_ddp.py \
      --train_dataset_dir outputs/training_sets/generated_training_sets/mixed_train_dataset_stride2 \
      --train_dataset_dir outputs/training_sets/real_world_merged_640x400_stride2 \
      --output_dir ./outputs/runs/new50mm_stride2 \
      --epochs 80 --batch_size 64 --train_ratio 0.9 \
      --amp --channels_last \
      --focal_loss --offset_head --size_head

cd /home/r13922171/dataset && uv run python src/train_yolo_detection.py --data outputs/datasets/yolo_detection/data.yaml --model yolo11n.pt --epochs 30 --imgsz 1024 --batch 80 --device 0,2 --workers 16 --cache ram
```
