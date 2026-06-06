# CCTag Synthetic Dataset Workspace

這個 repo 用來生成 CCTag 合成資料、檢查標註、訓練 heatmap regression 模型，並對模型做推論與評估。

  uv run torchrun --nproc_per_node=1 --master_port=29501 src/train_cctag_heatmap_ddp.py \
    --train_dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset_train \
    --train_dataset_dir /mnt/nvme2n1/r13922171/real_world_merged_640x400_train \
    --val_dataset_dir   ./outputs/training_sets/generated_training_sets/mixed_train_dataset_val \
    --val_dataset_dir   /mnt/nvme2n1/r13922171/real_world_merged_640x400_val \
    --output_dir ./outputs/runs/experiment_sizehead \
    --offset_head --size_head --focal_loss --batch_size 48

    
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

## Environment Management

這個 repo 現在以 `uv + pyproject.toml` 為主要依賴管理方式，並固定使用 Python `3.12.11`。

- `.python-version`：指定標準 Python 版本。
- `pyproject.toml`：唯一的主要依賴來源。
- `requirements/*.txt`：保留給舊流程或純 `pip` 使用的相容檔。
- `uv.lock`：鎖定實際解析出的完整依賴樹，讓不同機器重建出一致環境。


現在比較建議：
```bash

unset UV_INDEX_URL

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra cu126
```

然後把依賴需求維護在 `pyproject.toml`，把實際鎖定結果交給 `uv.lock`。

### 常用對照

- 建 venv + 安裝依賴：
  以前 `python -m venv .venv && pip install -r requirements.txt`
  現在 `uv sync --extra cu126`
- 重建一模一樣的環境：
  以前通常靠 `requirements.txt`
  現在靠 `uv.lock`
- 查看目前裝了什麼：
  以前 `pip freeze`
  現在可用 `uv pip freeze`

建議做法：

```bash
uv sync --extra cu126
```

如果這台機器沒有可用 CUDA，改用 CPU 版本：

```bash
uv sync --extra cpu
```

也可以用 bootstrap script：

```bash
bash scripts/bootstrap_env.sh
```

如果你還是要沿用 `pip`：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements/cu126.txt
```

或 CPU 版：

```bash
pip install -r requirements/cpu.txt
```

快速語法檢查：

```bash
uv run python -m py_compile \
  src/generate_cctag_dataset.py \
  src/visualize_random_labels.py \
  src/train_cctag_heatmap_ddp.py \
  src/infer_cctag_heatmap.py
```

## Generate Dataset

```bash
uv run python src/generate_cctag_dataset.py \
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
uv run python src/visualize_random_labels.py \
  --dataset_dir ./outputs/datasets/demo_small_testing \
  --num_samples 10 \
  --show_yolo_bbox \
  --output ./outputs/tmp/demo_small_testing_labels.jpg
```

## Train / Val / Test Split

現在推薦流程是：

- synthetic dataset 保持完整，直接當成 training source 之一。
- real-world dataset 要切成至少 `train` 和 `test`。
- 更建議切成 `train` / `val` / `test` 三份。
- training script 直接吃 multiple sources，不要再先把 synthetic 和 real-world physically merge 成一包。

### 為什麼不要把 `val` 和 `test` 用同一份

- `val` 是訓練中拿來選模型、調參數、看 overfitting 的。
- `test` 應該是最後才看的 held-out 評估集。
- 如果 `val == test`，你會不自覺對 test set overfit，最後分數會偏樂觀。

如果資料真的很少，短期內可以先用同一份資料做 `val/test`，但那一份就只能當 `val` 看，不能再把結果當成正式 test 結論。

### 推薦切法

以 real-world dataset 為主：

- `real_world_train`: 拿來訓練
- `real_world_val`: 拿來選 checkpoint / 調參
- `real_world_test`: 最後才評估

synthetic dataset 通常不需要再切 `test` 給最終報告；它主要是訓練來源。

### 用現有 split script 切資料

先從完整 real-world 切出 `train+val` 與 `test`：

```bash
uv run python scripts/split_train_test.py \
  --src outputs/real_world_stride4 \
  --train_dir outputs/real_world_stride4_trainval \
  --test_dir outputs/real_world_stride4_test \
  --test_ratio 0.15 \
  --seed 42
```

再把 `trainval` 繼續切成 `train` 與 `val`：

```bash
uv run python scripts/split_train_test.py \
  --src outputs/real_world_stride4_trainval \
  --train_dir outputs/real_world_stride4_train \
  --test_dir outputs/real_world_stride4_val \
  --test_ratio 0.15 \
  --seed 43
```

這樣大約會得到：

- `train`: 72.25%
- `val`: 12.75%
- `test`: 15%

## Train

訓練統一使用 DDP（`train_cctag_heatmap_ddp.py`）。

### 推薦做法：multi-source training

直接把 synthetic 和 real-world train split 一起丟進 training，不要先 merge：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
  --train_dataset_dir ./outputs/real_world_stride4_train \
  --val_dataset_dir ./outputs/real_world_stride4_val \
  --output_dir ./outputs/runs/experiment_multi_source_ddp \
  --epochs 80 \
  --batch_size 18
```

### 舊做法：單一 dataset + 內部分 train/val

如果你只有一包 dataset，還是可以沿用舊介面：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_mixed_ddp \
  --epochs 80 \
  --batch_size 18
```

但這種模式只適合 quick experiment，不適合拿來做正式的 real-world 評估。

small backbone：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --backbone mobilenet_v3_small \
  --train_dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
  --train_dataset_dir ./outputs/real_world_stride4_train \
  --val_dataset_dir ./outputs/real_world_stride4_val \
  --output_dir ./outputs/runs/experiment_mobilev3 \
  --epochs 80 \
  --batch_size 18
```

resnet：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --backbone resnet18 \
  --train_dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
  --train_dataset_dir ./outputs/real_world_stride4_train \
  --val_dataset_dir ./outputs/real_world_stride4_val \
  --output_dir ./outputs/runs/experiment_resnet18_ddp \
  --epochs 80 \
  --batch_size 18
```

### Focal Loss / OHEM

針對 false positive 問題，訓練時可啟用 Focal Loss 或 OHEM 來讓 model 專注在難分辨的 pixel 上。

**Focal Loss**（推薦）：對已經學會的 easy pixel（大片背景）自動降低 loss 權重，把學習力量集中在容易誤判的區域（過曝邊緣、曲線）。

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
  --train_dataset_dir ./outputs/real_world_stride4_train \
  --val_dataset_dir ./outputs/real_world_stride4_val \
  --output_dir ./outputs/runs/experiment_focal \
  --focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --epochs 80 \
  --batch_size 18
```

- `--focal_alpha`：正負樣本權重平衡。0.25 表示負樣本（背景）權重較高，讓 model 更認真學「什麼不是 CCTag」。
- `--focal_gamma`：focusing 強度。越大越忽略 easy pixel。預設 2.0 通常夠用。

**OHEM**（更激進的替代方案）：直接丟掉 easy pixel，只用 loss 最大的 K% pixel 來更新 model。

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --train_dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
  --train_dataset_dir ./outputs/real_world_stride4_train \
  --val_dataset_dir ./outputs/real_world_stride4_val \
  --output_dir ./outputs/runs/experiment_ohem \
  --ohem_ratio 0.3 \
  --epochs 80 \
  --batch_size 18
```

- `--ohem_ratio 0.3`：只保留最難的 30% pixel。不要跟 `--focal_loss` 同時用。

說明：

- 所有 checkpoint 與訓練紀錄都應輸出到 `outputs/runs/...`。

## Test / Inference

### 正式 test

訓練完成後，用 held-out `real_world_test` 做最終評估：

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_multi_source_ddp/best.pt \
  --input ./outputs/real_world_stride4_test/images \
  --dataset_dir ./outputs/real_world_stride4_test \
  --output ./outputs/inference/real_world_test_eval \
  --vis \
  --eval
```

### 單一資料夾推論 + evaluation

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_mixed_ddp/best.pt \
  --input ./outputs/real_world_stride4_test/images \
  --dataset_dir ./outputs/real_world_stride4_test \
  --output ./outputs/inference/results_ddp \
  --vis \
  --eval
```

單張圖片：

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_01/best.pt \
  --input ./assets/samples/cctag_reallife.png \
  --output ./outputs/inference/reallife_demo \
  --vis
```

### 即時 Tracking 模式（降低 False Positive）

用 `--tracking_mode` 一次開啟所有保守設定：

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_focal/best.pt \
  --input ./camera_feed \
  --tracking_mode
```

`--tracking_mode` 等同於 `--threshold 0.65 --min_peak_sharpness 3.0 --temporal_window 5`。也可以個別調整：

- `--min_peak_sharpness 3.0`：真的 CCTag 在 heatmap 上是尖銳的 Gaussian peak（sharpness > 3），過曝造成的 FP 是模糊的大面積 activation（sharpness < 2）。對推論速度幾乎無影響。
- `--temporal_window 5`：連續 5 幀中要有 3 幀偵測到才接受，過濾掉單幀跳動的 FP。
- `--threshold 0.65`：比預設 0.5 更保守的 peak 門檻。

## Batch Workflows

產生 training sets：

```bash
bash scripts/generate_training_sets.sh
```

切 dataset：

```bash
uv run python scripts/split_train_test.py --help
```

如果你想把 synthetic / real-world split 後再產生 combined train/test 目錄，也可以用：

```bash
uv run python scripts/build_combined_train_test.py --help
```

## Output Conventions

- `outputs/datasets/`: 一般資料集。
- `outputs/testing/`: 額外壓力測試集或 hard test suites。
- `outputs/training_sets/`: 由 workflow 腳本生成的訓練資料組合。
- `outputs/runs/`: checkpoint、training logs。
- `outputs/inference/`: heatmap、overlay、evaluation results。
- `outputs/tmp/`: 暫時檢查圖與其他中間產物。

舊的根目錄資料夾路徑已視為歷史結構，後續不再使用。
