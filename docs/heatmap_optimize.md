● 看完這個訓練 log，跟之前分析一致 — heatmap loss 收斂了但 center_l2 依然高且不穩定。

  但關鍵是：之前 v4 的 inference 測試結果實際是好的（center_l2 = 4-6px）。training 時的 center_l2 指標不準是因為 validation set 分佈不同、outlier 拉高均值。

  ---
  如果還想繼續優化，主要可以調的方向：

  1. 提高 heatmap 解析度（最直接）

  目前 80×50 → 1 heatmap pixel = 8px 圖像空間，精度天花板在那。改成 160×100 可以直接提升定位精度。

  2. 用 Focal Loss 取代 BCE

  BCE 對大量背景像素（都是 0）太敏感，模型傾向輸出 diffuse 的低值。Focal Loss 降低 easy negative 的梯度，讓模型專注學 peak。

  3. 縮小 heatmap 的 Gaussian sigma

  如果生成 heatmap 時的 sigma 太大，ground truth 本身就很 diffuse，模型學不出 sharp peak。

  4. 增加訓練資料

  7000 筆對於 EfficientNet-B0 偏少，加到 15-20k 會更穩定。

  ---
  但實際上，你之前 v4 inference 的表現已經很好了：


  ┌───────────────┬──────────────┬──────────────┐
  │    測試集     │ detection_f1 │ center_l2_px │
  ├───────────────┼──────────────┼──────────────┤
  │ boundary_hard │ 0.999        │ 6.3px        │
  ├───────────────┼──────────────┼──────────────┤
  │ extreme_mix   │ 0.991        │ 5.7px        │
  ├───────────────┼──────────────┼──────────────┤
  │ negative_hard │ 1.000        │ 5.1px        │
  ├───────────────┼──────────────┼──────────────┤
  │ small_hard    │ 0.993        │ 4.8px        │
  └───────────────┴──────────────┴──────────────┘

  除非你的應用需要 sub-pixel（<2px）精度，否則目前已經夠用了。要繼續推嗎？
---

## GPU vs CPU Inference Benchmark

**測試環境：**
- Model: CCTagNetV3 (EfficientNet-B0 backbone, U-Net skip connections)
- Input size: 640×400
- Test set: `small_testing` (100 張圖片)
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- 含 `--vis` overlay 輸出 + `--eval` 評估計算

**結果：**

| | GPU (CUDA) | CPU | 差異 |
|---|---|---|---|
| Wall time (real) | 6.98s | 13.27s | GPU 快 1.9× |
| User time | 38.4s | 4m53s | CPU 多核計算量大 |
| 每張圖片平均 | ~70ms | ~133ms | |

**Evaluation metrics（兩者完全一致）：**

| Metric | Value |
|---|---|
| detection_accuracy | 1.000000 |
| detection_f1 | 1.000000 |
| heatmap_pixel_f1 | 0.741708 |
| heatmap_pixel_iou | 0.589457 |
| mean_center_l2_px | 4.534743 |
| avg_combined_loss | 0.325522 |

**結論：**
- GPU 比 CPU 快約 1.9 倍（wall time），但差距不算特別大，因為 EfficientNet-B0 模型本身輕量
- CPU user time 遠高於 GPU（4m53s vs 38s），反映 CPU 需要大量多核並行計算
- 推論結果完全一致，數值精度無差異
- 若用更大模型或 batch inference，GPU 優勢會更顯著
