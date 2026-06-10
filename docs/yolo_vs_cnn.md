# YOLO Detection vs CNN Heatmap — CCTag 定位實驗完整紀錄

> 來源:2026-06-04 ~ 06-06 的工作 session(transcript 原檔在 `docs/history/`)
> 結論一句話:**經過「停整」(CNN 重訓到 1024 + offset/size head + 升級 decode)後,CNN heatmap 在真實 FLIR 圖上大幅領先 YOLO**;YOLO 適合當第一階段粗抓,CNN 適合第二階段精定位。

---

## 0. TL;DR

| 比較面向 | YOLO11n(detection) | CNN heatmap(resnet18) |
|---|---|---|
| 輸出 | bounding box(範圍/大小) | 次像素圓心 +(1024 版)size |
| precision(暗圖假框) | 低 — 暗圖噴 7–10 個假框,v2 重訓也沒根治 | **高 — 最暗 4 張 0 假框** |
| recall(大 marker) | 高但會切碎成多框 | 舊 640 會漏;**1024 重訓後修好** |
| 圓心精度 | 框中心(量化) | **次像素(offset head)** |
| GPU 速度(4K/18 tile) | 253ms(原始)/ 78ms(ONNX CUDA) | **58ms(compile+FP16)** |
| CPU 速度 | 1844ms(較快) | 4981ms(decoder 較重) |
| 生態 | ultralytics 一套(訓練/匯出/追蹤/mAP) | 手寫腳本,旋鈕都在自己手上 |
| **最終定位** | **stage-1 acquisition(粗抓不漏)** | **stage-2 localization(精修圓心)** |

兩者本質是 **同一條 precision/recall 取捨線上的兩個點**,不是誰絕對好。對「要精確圓心」的 CCTag fiducial,CNN 的「嚴格、置中、subpixel」是優點。

---

## 1. 背景與目標

- **原圖**:真實 FLIR 影像 4096×2160,marker(CCTag 同心圓 fiducial)距離 5m~40m 不等,場景偏暗、低對比。
- **需求**:盡量**保留 cctag 的像素數(px count)**,各距離都要穩定偵測。
- **動機**:嘗試用 YOLO 做 detection,評估能不能取代 / 搭配既有的 CNN heatmap 定位。

---

## 2. Session 1(2026-06-04)— YOLO 資料準備與訓練

### 2.1 資料集組建
用 `src/prepare_yolo_dataset.py` 把 6 個來源合併成 Ultralytics 格式:

| 來源 | 性質 |
|---|---|
| `det_positive_wide` | 正樣本(寬尺寸) |
| `det_far_small` | 遠距小 marker |
| `det_hard` | 難例 |
| `det_hard_negative` | 難負樣本(背景) |
| `det_overexposure` | 過曝(FP guard) |
| `real_world_merged_640x400` | 真實資料 |

- 合計 **19,547 張** → 17,592 train / 1,955 val(train_ratio 0.9)。
- 各來源加 `s{idx}_` 前綴避免 `000000` 編號碰撞。

### 2.2 踩到的坑
1. **輸出目錄沒清乾淨 → train/val 洩漏**
   - `prepare_yolo_dataset.py` 沒有在寫入前清空 `--output_dir`,重用舊目錄 + 不同 `--dataset_dir` 順序 → 殘留舊 run 檔案(不同 `s{idx}_` 前綴),同一張圖可能 train 一份、val 一份 → **驗證指標虛高**。
   - **修法**:`main()` 開頭 `shutil.rmtree` 掉 `images/`、`labels/` 子樹,每次乾淨重建。
2. **Ultralytics DDP 相對路徑 bug**
   - DDP 子進程會用相對路徑重啟腳本,改 CWD 後就壞掉。
   - **修法**:多 GPU 一律用**絕對路徑**(腳本、`--data`、`--project`)。
3. **GPU 沒吃滿(dataloader 瓶頸)**
   - nano 模型的瓶頸是 dataloader(PNG decode + augment),不是 GPU 算力;加上混到 A4000(GPU 4)是 straggler。
   - **修法**:`--cache ram`(~10GB,451GB RAM 綽綽有餘)+ batch 120 + workers 16 → GPU_mem 吃到 13.1G、~2 it/s。

### 2.3 訓練結果
- `yolo11n`,imgsz 1024,單類別 `cctag`。
- **mAP50 從 ~epoch 17 就 plateau 在 0.985**,mAP50-95 0.97~0.978。
- ⚠️ 當時就提醒:**這 0.985 是合成 val 的分數**(val 跟 train 同分佈),不代表真實場景。約 **20~25 epoch 就到頂**,patience 建議設 20~30。

---

## 3. Session 2(2026-06-05 ~ 06-06)— 真實圖驗證 + YOLO vs CNN

### 3.1 第一個災難:模型只訓練了 2 epoch
- 第一顆 `cctag_det_n` 因為訓練指令**換行/語法錯誤**(`--workers: command not found`)只跑了 2 epoch 就斷。
- 在真實 4K 暗圖上**噴 tile 大小的垃圾框、漏掉真 marker**,要把 conf 拉到 0.8 才壓得住 —— **需要 0.8 才能用本身就是模型有問題**。
- **教訓**:val mAP 0.98 漂亮但無用,是合成 val 的假象;真實圖是 OOD,欠訓練一遇 OOD 就崩。
- 重訓 30 epoch(`cctag_det_n_30ep`)後正常:真 marker 穩定 **0.97**,框大小正常貼合。

### 3.2 4K 推理策略:切 tile(`src/tile_detect.py`)
- 直接對整張 4K 偵測 → 內部縮成 1024 → 40m 小 marker 縮到 ~10px → 漏掉。
- **解法**:native-resolution sliding tile —— `tile=1024 + imgsz=1024`(1:1 不縮放)+ **30% overlap + NMS**。
  - overlap 確保 marker 不被切在接縫。
  - 4096×2160 → 6×3 = 18 個 tile。
- **tile vs imgsz 的關鍵**:切 tile 換取「有效解析度」,而不是去訓練一個 4K 模型。

### 3.3 input size / 4K 的設計討論
- **「train 一個 4K model」不合理**:合成訓練圖其實是 1024×540(4K 的 1/4 解析度),`imgsz=4096` 只是把它放大 4 倍餵進去,不會生出新細節,純浪費 16× 記憶體/時間。
- **最均衡的 YOLO input size = 1280**(stride 32 的倍數,小物件甜蜜點,P6 系列為此設計)。
- 但對「保留 px」需求,重點不是 imgsz,而是**切 tile 1:1**:
  - `tile=1280 + imgsz=1280` → 5×2 = 10 塊,比 1024 的 18 塊**更少、總像素更省、一樣 100% 保留 px**。
- **原則**:訓練時的 marker 像素大小要跟推理時一致(train/infer 尺度對齊)。

### 3.4 速度 benchmark(RTX 4070 Ti SUPER)

**GPU,切 18 tile 處理一張 4K:**

| 方法 | 每塊 | 一張 4K | FPS |
|---|---|---|---|
| **CNN heatmap + torch.compile + FP16** | 3.22 ms | **58 ms** | **17.2** ⚡ |
| CNN heatmap FP16 + channels_last | 5.27 ms | 95 ms | 10.5 |
| CNN heatmap FP32(原始) | 6.35 ms | 114 ms | 8.7 |
| YOLO11n ONNX Runtime CUDA(純推理,無 NMS) | 4.33 ms | **78 ms** | 12.8 |
| YOLO11n ultralytics predict(原始,含 NMS) | 14.0 ms | 253 ms | 4.0 |

**CPU(36 threads),切 18 tile:**

| 方法 | 每塊 | 一張 4K | FPS |
|---|---|---|---|
| YOLO11n @1024 | 102 ms | 1844 ms | 0.54 |
| CNN heatmap @1024 | 277 ms | 4981 ms | 0.20 |

**不切 tile(整張縮圖單次,快但丟小 marker):**

| 方法 | GPU | CPU |
|---|---|---|
| CNN heatmap @640×400 | 3.9 ms(258 FPS) | 65 ms(15 FPS) |
| YOLO @1024 單次 | 14 ms(71 FPS) | 102 ms(10 FPS) |

#### 速度結論(含反轉與但書)
1. **GPU + tile:CNN 較快**(優化後 58ms vs YOLO 253ms)。
2. **CPU 反轉:YOLO 較快 2.7×** —— resnet18 在 1024 的 FLOPs(~38G + 上採樣 decoder)其實比 yolo11n(~6.3G)重,GPU 平行度藏得住、CPU 藏不住。
3. **GPU/CPU 勝負相反**:GPU 選 CNN,CPU 選 YOLO;但兩者 CPU 都 <1 FPS,**4K tiled 必須 GPU**。
4. **YOLO 的 overhead 在 Python 前後處理 + NMS**(純 forward 只 ~2ms),所以 FP16 對 ultralytics predict 幾乎沒用;要快得走 **ONNX Runtime / TensorRT** 把 pipeline 編譯掉(實測 ORT CUDA 比 ultralytics predict 快 3.2×)。
5. **加速手段**:CNN 走 `torch.compile + FP16`(2.1~2.3×,但 ORT 反而變慢,因為上採樣 decoder 在 ORT 沒被優化);YOLO 走批次化 + TensorRT/ONNX。
   - TensorRT 當時 **build 失敗**(自動裝到 TRT 11,跟這版 ultralytics 的 `EXPLICIT_BATCH` 不相容),ONNX 有匯出。
   - ORT GPU 一開始偷偷 fallback 到 CPU(缺 `libcudnn.so.9`),其實 cuDNN 9.10 已在 venv 內,只是不在 `LD_LIBRARY_PATH` → 加 `nvidia/*/lib` 路徑即修好。

### 3.5 準度對比(21 張真實 FLIR)— **中段快照(舊 640 CNN vs 舊 YOLO)**

> ⚠️ 這是「停整之前」的舊表,後面 §4 才是修正後的結論。

| 情境 | YOLO | CNN(舊 640) | 較好 |
|---|---|---|---|
| 5m 近距大 marker | 2(碎框) | **0(漏)** | YOLO |
| dichoric_10m | 3 | **0(漏)** | YOLO |
| 暗圖光圈(6 張) | **7–10 個假框** | 1 | CNN |
| 40m 遠距小 marker | 真框 + 1 FP | score 1.00,零假框 | CNN |

- **YOLO = 高 recall、低 precision**(少漏抓但暗圖噴假框)。
- **CNN = 高 precision、低 recall**(超乾淨但大 marker / 某些圖漏抓)。

#### 為什麼性格相反(同一張 5m)
| | CNN heatmap | YOLO |
|---|---|---|
| 觸發條件 | 要**看到完整、置中的同心圓心**才有峰值 | 看到**夠多圈**就能咬一個框 |
| 大 marker 跨 tile | tile 只看到外圈幾條弧 → 沒中心 → **沒峰值 → 漏** | 看到幾圈 → **不漏,但碎框** |
| 暗圖亮點 | 門檻嚴 → **假框少** | 夠像圈就咬 → **假框多** |

→ 「高 recall 低 precision」vs「高 precision 低 recall」不是兩個模型好壞,是 **同一條 precision/recall 取捨線上的兩個點,沒有免費的午餐**。

---

## 4. 「停整」—— 讓 CNN 反超的關鍵(2026-06-05 下午 → 06-06)

針對舊 640 CNN 唯一的弱點(大 marker 漏抓 + 圓心量化),做了四件事:

1. **CNN 重訓到 1024×640**
   - 對齊 tile 部署尺度,修掉 **train(640)/infer(1024 tile)尺度不一致**(當時 CNN 最大的隱性問題)。
   - checkpoint:`outputs/runs/heatmap_1024/best.pt`。
2. **offset head → 次像素圓心**
   - heatmap 峰值落在 stride-4 格點(最多差 4px),offset head 預測「真圓心相對格點的偏移」→ 加上去得次像素精度。
3. **size head + sharpness 濾假框**
   - size head 預測 marker 大小,亮邊 FP 的預測尺寸不合理 → 濾掉;`min_peak_sharpness` 拒絕不夠尖/拉長的鈍峰。
4. **多尺度 tiling**
   - 除了 native pass,再加縮小 pass,讓 5m 的 ~1025px 大 marker 縮回 tile 內,圓心才有峰值。

### 4.1 升級 decode 效果(basic vs upgraded,峰數)

| 圖 | basic | upgraded | |
|---|---|---|---|
| 40m_example | 4 | **2** | FP↓(size head 濾掉 4 個亮邊 FP) |
| galvo_10m | 4 | **1** | FP↓ |
| 10m 光圈 | 3 | **1** | FP↓ |
| d.and.g_10m | 3 | **2** | FP↓ |
| 最暗 4 張(214641-643, 40m 光圈) | 0 | **0** | 乾淨 ✓ |
| 5m | 1 | 1 | 真 marker ✓ |

### 4.2 修正後的最終對比(舊 640 → 停整後 1024)

| 情境 | 舊 640 CNN | **停整後 1024 CNN** |
|---|---|---|
| 5m 近距大 marker | 0(漏) | ✅ 抓到(1024 尺度對齊修好) |
| dichoric_10m | 0(漏) | ✅ 抓到,FP 4→1 |
| 暗圖光圈(6 張) | 1(已乾淨) | ✅ 維持乾淨 |
| 40m 遠距小 marker | score 1.00,零假框 | ✅ 次像素圓心 + size 框 |
| 最暗 4 張無-marker | — | ✅ **0 峰**(YOLO 在這噴一堆) |
| 40m_example 亮邊 FP | — | 4 → **2**(size head 濾掉) |

**結論**:新 1024 CNN(offset+size head)+ 升級 decode → **主目標(5m/10m 漏抓 + 圓心精度)達成,最暗無-marker 圖 0 假框,遠勝 YOLO**。

### 4.3 為什麼「CNN 效果好很多」
不是 CNN 架構天生贏,而是:
- CNN 的 **precision 本來就高**(嚴格、要看到完整圓心)→ 暗圖幾乎零假框,YOLO 一直噴 7–10 個。
- 停整補上了它**唯一的弱點**(大 marker 漏抓)→ recall 也補回來。
- 額外拿到**次像素圓心**,正好是 CCTag fiducial 最需要的精度。

YOLO 即使重訓 v2(`cctag_det_n_v2`,marker 尺寸放大 + 更暗 low-light),**暗圖假框仍沒根治**(沒加真實 hard negative),最高假框分數到 0.82,光提 conf 清不掉。

---

## 5. 殘留問題與後續方向

| 問題 | 解法 | 狀態 |
|---|---|---|
| 同一大 marker 被偵測兩次(去重半徑寫死 25px 對大 marker 太小) | 去重半徑**隨 marker 大小縮放**(用 size head 半徑 × 0.5) | 簡單可修,待做 |
| 結構邊緣 FP(亮/暗門框角落) | 真實 **hard negative**(裁 21 張 FLIR 無-marker 區)混進訓練 | 要拍/裁資料 |
| YOLO 暗圖假框 | 快解:conf 0.65~0.75;正解:真實 hard negative | 快解可用 |
| CPU 不可用於 4K tiled(<1 FPS) | 必須 GPU;或粗定位→局部精修減少 tile 數 | 架構選擇 |

### two-stage 建議
> **YOLO 粗抓(acquisition,不漏)→ CNN 精修(localization,subpixel 圓心)** —— 正是 repo 註解寫的 "two-stage localization" 的道理。
> 或:整張縮圖 CNN(3.9ms)當粗定位 → 只在熱區切 native tile 精抓,把 18 塊降到 1~3 塊,潛在 30~60 FPS。

---

## 6. 產出物 / 關鍵路徑

| 項目 | 路徑 |
|---|---|
| YOLO tiled 推理腳本 | `src/tile_detect.py` |
| CNN tiled 推理腳本(升級 decode) | `src/tile_heatmap.py` |
| YOLO 權重(30ep) | `outputs/runs_yolo/cctag_det_n_30ep/weights/best.pt` |
| YOLO 權重(v2) | `outputs/runs_yolo/cctag_det_n_v2/weights/best.pt` |
| CNN 1024 權重 | `outputs/runs/heatmap_1024/best.pt` |
| CNN 舊 640 權重 | `outputs/runs/stride4_offw1.0/best.pt` |
| CNN 1024 升級 decode overlay | `outputs/inference/hm1024_upgraded/` |
| 推理指令 | `command.md`(INFERENCE 區塊) |
| 原始 transcript | `docs/history/yolo_session1_2026-06-04_dataset-and-training.txt`<br>`docs/history/yolo_session2_2026-06-05_yolo-vs-cnn-comparison.txt` |

### 現行推理指令(摘自 command.md)
```bash
# YOLO 偵測(tiled)→ bounding box
CUDA_VISIBLE_DEVICES=1 uv run python src/tile_detect.py \
  --model outputs/runs_yolo/cctag_det_n_v2/weights/best.pt \
  --source 40m_example.png --output outputs/runs_yolo/flir_v2_c07 \
  --imgsz 1024 --tile 1024 --overlap 0.3 --conf 0.7 --iou 0.5

# CNN heatmap(tiled,升級 decode)→ 次像素圓心 + size
CUDA_VISIBLE_DEVICES=0 uv run python src/tile_heatmap.py \
  --checkpoint outputs/runs/heatmap_1024/best.pt \
  --source 40m_example.png --output outputs/inference/hm1024_upgraded \
  --threshold 0.5 --min_peak_sharpness 3.0 --max_size_frac 1.2
```
