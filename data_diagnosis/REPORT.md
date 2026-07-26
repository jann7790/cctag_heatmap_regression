# CCTagNet 資料切分與泛化診斷報告

> 範圍:**只修評估 + 診斷資料**。未訓練模型、未更動 backbone/架構,未覆寫任何原始資料或現有切分。所有產出在 `data_diagnosis/`。
> 診斷對象:`fable_occ_1024_offset` 訓練設定,真實資料 = `6f_labeled_1024x640_roi`(+`_occ`),來源 = `6f_labeled`(1073 幀,4096×2160,CCTag lib 自動標註)。

---

## TL;DR

1. **現有隨機切分嚴重洩漏:** 落在 val 的真實幀,**100%** 其「完全相同的來源幀」也在 train(不是相鄰 1~2 幀,是距離 0)。原因:同一來源幀被炸成 ~9–12 個近重複 crop(`pos/neg/occ`),全域隨機 shuffle 把它們灑到兩邊。**現有 val 量的是「同幀記憶」,不是泛化。** → train/val 沒 gap 是必然。
2. **「遠距 label 又少又差」假設被推翻:** 在有標註的距離範圍內,遠/小 marker 偵測率 94–100%、標註中心精準。266 個漏標(no_detection)其實集中在**幾個暗光/模糊的連續區塊**(尤其結尾 114 幀全黑 tail),是相機條件問題,不是遠距問題。
3. **已產出無洩漏的新切分:** 用 marker 半徑當距離軸,把最遠段(r<92px,142 幀)整段抽成 `real_holdout`,以來源幀分組 + 時間 buffer,驗證 train↔holdout 幀 0 重疊、最小時間間隔 4 幀。
4. **下一步判斷:** 先用新 holdout 量出誠實的遠距數字,**再**談模型。資料端唯一要修的是「**極遠距離的涵蓋範圍/蒐集**」(holdout 最遠僅 r≈59px,是否真達 40m 無法確認),**不是**標註品質。模型/backbone 先不要動。

---

## 任務一:現有切分的洩漏量化

精確重現 `split_indices`(seed=42 全域 shuffle 26067 樣本 → 切 10%=2607 val),篩出真實 ROI crop:

| 最近 train 鄰居 frame-index 距離 | 真實 val crop 比例 |
|---|---|
| ≤ 0 幀(同一來源幀已在 train) | **996 / 996 = 100.0%** |
| ≤ 1 / ≤ 2 / ≤ 5 / ≤ 10 幀 | 100.0%(皆同上) |

**洩漏機制有兩層:**(a) 相鄰秒幀近乎相同;(b) 更直接 —— 同一幀的 pos0–3 / neg0–2 / occ 變體是近重複,被隨機灑到 train/val 兩邊。

- 圖:`task1_leakage_hist.png`(單一柱全壓在距離 0)
- 文字:`task1_leakage_summary.txt`
- 重現腳本:`task1_leakage.py`

---

## 任務二:CCTag 標註品質 vs 距離

距離 proxy = 來源全幅 marker 半徑 `r = 0.5*(ellipse_a+ellipse_b)`(使用者確認時間≠單調距離,故改用 marker 大小)。

**偵測率 vs 距離(誠實版:detected + 孤立漏標;區塊 dropout 因距離不明而排除):**

| 半徑 px | 幀數 | 偵測率 | |
|---|---|---|---|
| [0,80) 遠 | 26 | 1.00 | |
| [80,120) | 392 | 0.94 | |
| [120,160) | 109 | 0.96 | |
| [160,220) | 99 | 0.91 | |
| [220,300) | 83 | 0.87 | |
| [300,450) | 84 | 0.90 | |
| [450,700) | 87 | 0.78 | |
| [700,1100) 近 | 9 | 0.33 | ← n 極小,全是開頭模糊架設幀 |

**漏標真正成因 = 連續區塊 dropout(非遠距):** 266 個漏標只有 82 個離某成功偵測 ≤3 幀;其餘 184 個在「整段沒偵測到」的區塊:

| 起始 rank | 長度 | 時間段 | 看圖判讀 |
|---|---|---|---|
| 959 | **114** | 22:49:11→22:51:04(結尾) | 全黑/低光,marker 不在視野 |
| 0 | 27 | 開頭 | 架設中,近全黑 + 運動模糊 |
| 902 | 23 | 22:48 | 暗段 |
| 522 | 18 | 近距波峰 | 近距快速移動 → 動態模糊 |

**抽樣肉眼檢查**(`task2_samples/`):
- 近 `near_r696..1032_*`、中 `mid_r122_*`、遠 `far_r58..73_*`:遠距標註中心十字精準。
- 漏標 `missed_*`(全幅縮圖):確認是暗/模糊/沒對準 marker 的畫面,非「小遠 marker」。

- 圖:`task2_radius_vs_time.png`(來回 4 週期軌跡 + 漏標位置)、`task2_detection_rate_vs_distance.png`、`task2_label_density.png`
- 文字:`task2_summary.txt`;腳本:`task2_label_quality.py`

> ⚠️ 尺度未知數:半徑波谷最遠僅 r≈59px。因來回移動 + 無相機標定,**此半徑是否對應 40m 無法確認**。

---

## 任務三:無洩漏的距離段重切

原則:① 真實資料按距離(半徑)切,最遠段當 holdout;② 以**來源幀分組**(同幀所有 crop 同 bucket);③ **時間 buffer**(丟掉時間上離 holdout ≤3 幀的非 holdout 幀);④ 合成資料隨機 10% 當 dev val(僅供 early-stopping/選 checkpoint,**非泛化成績**)。

| Bucket | 真實來源幀 | crop 數 | 半徑範圍 (px) | 清單檔 |
|---|---|---|---|---|
| **real_holdout**(遠,訓練全程不碰) | 142 | 2020 | 58.9 – 92.0 | `split_lists/real_holdout.txt` |
| real train(近+中) | 855 | — | 92.0 – 1032 | (含於 `train.txt`) |
| buffer(丟棄) | 76 | 916 | 邊界 | `split_lists/real_buffer_dropped.txt` |
| synthetic dev_val | — | 1350 | — | `split_lists/synthetic_dev_val.txt` |
| **train.txt**(全部訓練) | — | 21781 | — | `split_lists/train.txt` |

**洩漏驗證(全部通過):**
- train ∩ holdout 來源幀 = **0**(切斷任務一的近重複洩漏)
- train / holdout 半徑無重疊(<92 | ≥92)
- train↔holdout 最小時間間隔 = **4 幀**(>buffer_K=3;幀距~2s → ≈8s)
- 對比:舊切分 100% val 幀來源在 train → 新 holdout 0%。

> holdout 嚴格比 train 所有幀都更遠(train 最小 r=92px,holdout 全 <92px)→ 真正測「遠距外推」。
> 旋鈕:`R_LO`(調更小=更遠 holdout)、`BUFFER_K`、`DEV_VAL_FRAC`。腳本 `task3_resplit.py`;統計 `task3_manifest.json`;圖 `task3_distance_split.png`。

---

## 下一步:先修資料,還是先動模型?

**順序建議:先把評估量誠實 → 量數字 → 再決定。模型/backbone 先不要動。**

1. **(最優先,零成本)用新 `real_holdout.txt` 重新評估現有最佳 checkpoint。**
   不重訓,直接量遠距(r<92px,且嚴格比 train 更遠)的 center-L2 / 偵測率。這才是第一個誠實的泛化數字。在它出來前,任何換 backbone 都是猜測(與你原本判斷一致)。

2. **資料端唯一該修的是「涵蓋範圍」,不是標註品質。**
   任務二證明遠距 label 既不稀疏也不偏 → 不需要重標。但 holdout 最遠僅 r≈59px,**若部署要的 40m 比資料最遠端更遠,就是資料沒涵蓋到** → 需補拍更遠的真實片段(這是資料工作,但與「標註品質」無關)。另可清掉結尾 114 幀全黑 tail 之類的無效負樣本區塊。

3. **用無洩漏切分重訓後,再看 dev_val(選 checkpoint)與 holdout(報泛化)的 gap。**
   - 若 holdout 表現明顯差於舊 val:證實先前「沒 gap」是洩漏假象,泛化確實弱 → 此時才有依據考慮模型容量/解析度。
   - 若 holdout 表現其實不差:那「遠距變差」更可能是**資料涵蓋不到真正的 40m**,該補資料而非換模型。

**一句話:** 瓶頸不是標註品質(已排除),而是 (a) 評估方法(本次已修)與 (b) 極遠距離的資料涵蓋。先量誠實數字,模型決策留到有 holdout 成績之後。

---

## 產出檔案索引(皆在 `data_diagnosis/`)

```
task1_leakage.py / _summary.txt / _hist.png
task2_label_quality.py / _summary.txt
  task2_radius_vs_time.png / task2_detection_rate_vs_distance.png / task2_label_density.png
  task2_samples/  (near_* mid_* far_* missed_*)
task3_resplit.py / task3_manifest.json / task3_distance_split.png
  split_lists/  train.txt  synthetic_dev_val.txt  real_holdout.txt  real_buffer_dropped.txt
REPORT.md  (本檔)
```
