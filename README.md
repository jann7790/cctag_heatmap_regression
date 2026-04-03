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

## Environment Management

這個 repo 現在以 `uv + pyproject.toml` 為主要依賴管理方式，並固定使用 Python `3.12.11`。

- `.python-version`：指定標準 Python 版本。
- `pyproject.toml`：唯一的主要依賴來源。
- `requirements/*.txt`：保留給舊流程或純 `pip` 使用的相容檔。
- `uv.lock`：鎖定實際解析出的完整依賴樹，讓不同機器重建出一致環境。

### 如果你以前都用 requirements.txt

可以先用這個心智模型理解：

- `requirements.txt`：你手寫「想裝什麼」。
- `pip freeze`：把「目前環境實際裝了什麼」全部列出來。
- `pyproject.toml`：新版的主要依賴定義檔，比 `requirements.txt` 更完整。
- `uv.lock`：比 `pip freeze` 更適合提交到 repo 的鎖檔。
- `uv sync`：依照 `pyproject.toml` 和 `uv.lock` 建立一致環境。

`pip freeze` 會輸出目前 virtualenv 內已安裝套件的精確版本，例如：

```bash
pip freeze
```

可能會得到：

```text
numpy==2.4.3
opencv-python==4.13.0.92
torch==2.11.0+cu126
torchvision==0.26.0+cu126
```

但它有兩個限制：

- 它描述的是「現在這個環境碰巧裝了什麼」，不一定是你真正想維護的最小依賴集合。
- 它常會把很多間接依賴一起寫出來，久了會很難整理。

以前常見流程是：

```bash
pip install -r requirements.txt
pip freeze > requirements.txt
```

現在比較建議：

```bash
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

## Train

訓練統一使用 DDP（`train_cctag_heatmap_ddp.py`）：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_mixed_ddp \
  --epochs 80 \
  --batch_size 18
```

small backbone：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --backbone mobilenet_v3_small \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_mobilev3 \
  --epochs 80 \
  --batch_size 18
```

resnet：

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_resnet18_ddp \
  --backbone resnet18 \
  --epochs 80 \
  --batch_size 18
```

### Focal Loss / OHEM

針對 false positive 問題，訓練時可啟用 Focal Loss 或 OHEM 來讓 model 專注在難分辨的 pixel 上。

**Focal Loss**（推薦）：對已經學會的 easy pixel（大片背景）自動降低 loss 權重，把學習力量集中在容易誤判的區域（過曝邊緣、曲線）。

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
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
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_ohem \
  --ohem_ratio 0.3 \
  --epochs 80 \
  --batch_size 18
```

- `--ohem_ratio 0.3`：只保留最難的 30% pixel。不要跟 `--focal_loss` 同時用。

說明：

- 所有 checkpoint 與訓練紀錄都應輸出到 `outputs/runs/...`。

## Inference

資料夾推論 + heatmap + overlay + evaluation：

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_mixed_ddp/best.pt \
  --input ./outputs/testing/small_testing/images \
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
