# False Positive 降低策略

## 問題背景

這個 model 用於即時 camera tracking，將 CCTag 保持在畫面正中央。  
主要 false positive 場景：**CCTag 不在畫面中，但過曝區域加上曲線、弧形紋理觸發偵測**，導致鏡頭跑掉。

根本原因是訓練資料缺乏這類場景，model 從來沒見過「沒有 CCTag 但有過曝曲線」的負樣本，不知道該輸出零 heatmap。

---

## 對策一：Training Loss — Focal Loss 與 OHEM

### 為什麼 BCE 不夠好

標準 BCE 對每個 pixel 一視同仁。但 heatmap 是 80×50 = 4000 個 pixel，其中真正的 Gaussian peak 只佔約 50 個 pixel。

- **99% 的 pixel 是背景**（target = 0）
- 這些 easy negative pixel 每個 loss 很小，但數量龐大，加起來主導整個 gradient
- 結果：model 把大部分學習容量花在「確認背景是背景」，而不是學「過曝的曲線不是 CCTag」

---

### Focal Loss

**原理：** 加入一個 modulating factor `(1 - pt)^γ`，根據 model 預測的確信度動態調整 loss 權重。

```
FL(p, t) = -α · (1 - pt)^γ · log(pt)

pt = model 預測正確的機率（答對越有把握，pt 越大）
γ  = focusing parameter（越大越忽略 easy pixel，預設 2.0）
α  = 正負樣本平衡權重（預設 0.25）
```

**直觀解釋：**

| 場景 | pt | (1-pt)² | 效果 |
|------|----|---------|------|
| 一般背景 pixel，model 很確定 → 預測 0.02 | 0.98 | 0.0004 | loss 被壓到幾乎為零 |
| 過曝曲線，model 不確定 → 預測 0.4 | 0.6 | 0.16 | loss 保留 16% |
| 過曝曲線，model 搞錯 → 預測 0.7 | 0.3 | 0.49 | **loss 放大，強迫 model 學** |

model 已經會的東西不再浪費 gradient，所有學習力量集中在「搞不清楚的 pixel」上。

**`α = 0.25` 的含義：** 正樣本（CCTag peak）loss 權重 0.25，負樣本（背景）loss 權重 0.75。讓 model 更認真學「什麼不是 CCTag」。

**使用方式：**

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_focal \
  --focal_loss \
  --focal_gamma 2.0 \
  --focal_alpha 0.25 \
  --epochs 80 \
  --batch_size 18
```

**參數調整建議：**

- `--focal_gamma 2.0`：預設值，通常夠用
- `--focal_gamma 3.0`：更激進地忽略 easy pixel，適合 negative 樣本很多的資料集
- `--focal_alpha 0.25`：正負不平衡嚴重時可調低（更重視負樣本）

---

### OHEM（Online Hard Example Mining）

**原理：** 直接丟掉 easy pixel，只用 loss 最大的 top-K% pixel 來更新 model。

```
1. 算出每個 pixel 的 BCE loss（不平均）
2. 排序，只保留最大的 top-K%
3. 只用這些 pixel 的 loss 計算 gradient
```

**與 Focal Loss 的比較：**

|  | Focal Loss | OHEM |
|--|-----------|------|
| 做法 | soft：連續權重衰減 | hard：直接丟掉 easy pixel |
| easy pixel | loss 被壓小但還存在 | 完全不參與 gradient |
| 穩定性 | 較穩定 | loss 曲線有時跳動較大 |
| 建議 | 優先嘗試 | focal loss 不夠時再試 |

**使用方式（不要跟 `--focal_loss` 同時開）：**

```bash
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
  --output_dir ./outputs/runs/experiment_ohem \
  --ohem_ratio 0.3 \
  --epochs 80 \
  --batch_size 18
```

- `--ohem_ratio 0.3`：只保留最難的 30% pixel（約 1200 / 4000 個）
- 值越小越激進，但太小可能造成 loss 不穩定（建議 0.2 ~ 0.4）

**建議策略：先跑 focal loss，FP 還是高再試 OHEM。**

---

## 對策二：Inference 端 FP 過濾

這三個機制在推論時過濾掉 false positive，對推論速度幾乎無影響（< 0.01ms，相比 model forward pass 的 2-10ms 可忽略）。

---

### Peak Sharpness Check

**原理：** 真的 CCTag 在 heatmap 上會產生一個尖銳的 Gaussian peak。過曝造成的 FP 是模糊的大面積 activation。

量化方式：`sharpness = peak_value / mean(surrounding_region)`

| 偵測類型 | sharpness 典型值 |
|---------|----------------|
| 真的 CCTag（sharp Gaussian） | > 3.0 |
| 過曝 FP（diffuse activation） | < 2.0 |
| 邊界模糊 | 2.0 ~ 3.0（需調整門檻） |

**使用方式：**

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint best.pt \
  --input ./images \
  --min_peak_sharpness 3.0
```

- `--min_peak_sharpness 0.0`：停用（預設）
- `--min_peak_sharpness 3.0`：推薦起始值
- 如果 true positive 被誤砍，調低到 2.0；如果 FP 還是過多，調高到 4.0

---

### Temporal Consistency Filter

**原理：** 真實偵測在連續幀之間是穩定的。FP 通常是單幀跳動（一幀有、下幀沒有）。

實作：ring buffer 記錄最近 N 幀的偵測結果，要求 N×60% 以上的幀有偵測才接受。  
例如 `--temporal_window 5`：5 幀中要有至少 3 幀偵測到才觸發。

**使用方式：**

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint best.pt \
  --input ./images \
  --temporal_window 5
```

- `--temporal_window 0`：停用（預設）
- `--temporal_window 3`：較少延遲，輕度過濾
- `--temporal_window 5`：推薦，約 5 幀延遲但有效過濾單幀 FP

**注意：** temporal filter 會在 CCTag 剛進入畫面時有幾幀延遲才開始偵測，這是正常的 trade-off。

---

### Threshold 調整

預設 `--threshold 0.5`。對 tracking 用途建議調高：

- `--threshold 0.65`：保守，適合 FP 問題嚴重的場景
- `--threshold 0.75`：非常保守，CCTag 必須在清晰的狀態才偵測得到

---

### Tracking Mode（一鍵開啟）

`--tracking_mode` 自動設定所有保守預設值：

| 參數 | tracking_mode 預設 | 一般預設 |
|------|--------------------|---------|
| `--threshold` | 0.65 | 0.5 |
| `--min_peak_sharpness` | 3.0 | 0.0（停用） |
| `--temporal_window` | 5 | 0（停用） |

```bash
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_focal/best.pt \
  --input ./camera_feed \
  --tracking_mode
```

若需要個別覆蓋，在 `--tracking_mode` 後面再加參數即可：

```bash
# tracking_mode 但 sharpness 門檻調低（model 還沒重訓，peak 較不尖銳）
uv run python src/infer_cctag_heatmap.py \
  --checkpoint best.pt \
  --input ./camera_feed \
  --tracking_mode \
  --min_peak_sharpness 2.0
```

---

## 推薦的完整流程

```bash
# 1. 生成含 hard negative 和過曝場景的訓練資料
bash scripts/generate_training_sets.sh

# 2. 合併並訓練（已預設開 focal_loss）
bash scripts/generate_and_train_mixed.sh

# 3. 生成含 FP 壓力測試的測試集
bash scripts/generate_testing_suites.sh

# 4. Benchmark（確認 FP 有改善）
uv run python src/benchmark.py \
  --runs_dir ./outputs/runs \
  --suites_dir ./outputs/testing

# 5. 即時 tracking 推論
uv run python src/infer_cctag_heatmap.py \
  --checkpoint ./outputs/runs/experiment_focal/best.pt \
  --input ./camera_feed \
  --tracking_mode
```

---

## 各對策的效果預期

| 對策 | 主要效果 | 需要重訓？ |
|------|---------|-----------|
| Hard negative data | 讓 model 學到「背景曲線不是 CCTag」 | 是 |
| Overexposure data | 讓 model 學到「過曝場景怎麼輸出零」 | 是 |
| Focal Loss | 把學習力集中在難分辨的 pixel | 是 |
| Peak sharpness | 直接過濾掉模糊的 FP activation | 否，立即生效 |
| Temporal filter | 過濾掉單幀跳動的 FP | 否，立即生效 |
| Threshold 提高 | 最簡單，但會犧牲部分 TP 召回率 | 否，立即生效 |

**短期（不想重訓）：** 先加 `--tracking_mode`，立即降低 FP。  
**中長期（根本解決）：** 用新的訓練資料 + `--focal_loss` 重訓。
