# CCTag Synthetic Dataset Workspace

這個 repo 用來生成 CCTag 合成資料、檢查標註、訓練 heatmap regression 模型，並對模型做推論與評估。

## Repository Layout

```text
.
├── src/                    # Python entry points
├── scripts/                # Shell workflows
├── docs/                   # Notes and repo-planning docs
├── assets/
│   ├── markers/            # Reference CCTag assets
│   └── samples/            # Sample images kept in repo
└── outputs/                # All local generated artifacts
    ├── datasets/
    ├── testing/
    ├── training_sets/
    ├── runs/
    ├── inference/
    └── tmp/
```

規則很簡單：

- 正式內容放在 `src/`、`scripts/`、`docs/`、`assets/`。
- 本機生成內容只放在 `outputs/`。
- 根目錄不再直接放 `*_dataset/`、`runs/`、`results_*` 這類工作產物。

## Setup

使用 Python 3.11+，先安裝依賴：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

快速語法檢查：

```bash
python -m py_compile \
  src/generate_cctag_dataset.py \
  src/visualize_random_labels.py \
  src/train_cctag_heatmap.py \
  src/train_cctag_heatmap_ddp.py \
  src/infer_cctag_heatmap.py
```

## Generate Dataset

```bash
python src/generate_cctag_dataset.py \
  --num_images 100 \
  --output_dir ./outputs/datasets/demo_small_testing \
  --output_size 640x400 \
  --seed 777 \
  --marker_min 66 \
  --marker_max 333 \
  --partial_out_max_ratio 0.25 \
  --occ_min 0.85 \
  --occ_max 0.98 \
  --soft_focus_strength 0.9 \
  --empty_negative_ratio 0.15 \
  --boundary_target_ratio 0.20 \
  --visualize
```

生成後資料夾會包含：

- `images/`
- `heatmaps/`
- `labels_yolo/`
- `labels.csv`
- `config.json`

## Visualize Labels

```bash
python src/visualize_random_labels.py \
  --dataset_dir ./outputs/datasets/demo_small_testing \
  --num_samples 10 \
  --show_yolo_bbox \
  --output ./outputs/tmp/demo_small_testing_labels.jpg
```

## Train

單機 DataParallel：

```bash
python src/train_cctag_heatmap.py \
  --dataset_dir ./outputs/datasets/ultimate_dataset \
  --output_dir ./outputs/runs/experiment_03 \
  --epochs 80 \
  --batch_size 72 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --train_ratio 0.9 \
  --num_workers 8 \
  --seed 42 \
  --input_width 640 \
  --input_height 400 \
  --save_every 10 \
  --gpus 0,1,2
```

DDP：

```bash
torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_mixed_ddp \
  --epochs 80 \
  --batch_size 18
```

說明：

- `--gpus 0,1,2` 用於 `src/train_cctag_heatmap.py` 的 DataParallel。
- 所有 checkpoint 與訓練紀錄都應輸出到 `outputs/runs/...`。

## Inference

資料夾推論 + heatmap + overlay + evaluation：

```bash
python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_mixed_ddp/best.pt \
  --input ./outputs/testing/small_testing/images \
  --output ./outputs/inference/results_ddp \
  --vis \
  --eval
```

另一個 checkpoint：

```bash
python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_mixed_ddp/epoch_010.pt \
  --input ./outputs/testing/small_testing/images \
  --output ./outputs/inference/results_ddp_010 \
  --vis \
  --eval
```

單張圖片：

```bash
python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_01/best.pt \
  --input ./assets/samples/cctag_reallife.png \
  --output ./outputs/inference/reallife_demo \
  --vis
```

## Batch Workflows

產生 training sets：

```bash
bash scripts/generate_training_sets.sh
```

產生 testing suites：

```bash
bash scripts/generate_testing_suites.sh
```

產生 mixed training set 並直接開訓：

```bash
bash scripts/generate_and_train_mixed.sh
```

## Output Conventions

- `outputs/datasets/`: 一般資料集。
- `outputs/testing/`: 壓力測試集與 testing suites。
- `outputs/training_sets/`: 由 workflow 腳本生成的訓練資料組合。
- `outputs/runs/`: checkpoint、training logs。
- `outputs/inference/`: heatmap、overlay、evaluation results。
- `outputs/tmp/`: 暫時檢查圖與其他中間產物。

舊的根目錄資料夾路徑已視為歷史結構，後續不再使用。
