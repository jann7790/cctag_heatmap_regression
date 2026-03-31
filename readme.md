## generate
python generate_cctag_dataset.py \
   --num_images 100 \
   --output_dir ./small_testing \
   --output_size 640x400 \
   --seed 777 \
   --marker_min 66 \
   --marker_max 333 \
   --partial_out_max_ratio 0.25 \
   --occ_min 0.85 \
   --occ_max 0.98 \
   --soft_focus_strength 0.9 \
   --empty_negative_ratio 0.15 \
   --boundary_target_ratio 0.20  

## train

 python train_cctag_heatmap.py \
    --dataset_dir ./ultimate_dataset \
    --output_dir ./runs/experiment_03 \
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
    --coord_loss_weight 0.1 \
    --gpus 0,1,2
  注意事項：
  - --gpus 0,1,2 取代原本的 --device cuda，會自動以 GPU 0 為主卡
  - batch_size 16 是總 batch，DataParallel 會自動分成每張 GPU 各 ~5~6 筆
  - 如果想加速，可以把 batch_size 調大（例如 48），充分利用 3 張 GPU
  - 儲存的 checkpoint 已正確解包 model.module，載入時不需要額外處理

  
## infer
  
  整個資料夾 + 儲存 heatmap + 視覺化overlay：
  python infer_cctag_heatmap.py \
      --checkpoint ./runs/experiment_mixed_ddp/best.pt \
      --input small_testing/images \
      --output ./results_ddp/ \
      --vis \
      --eval


  python infer_cctag_heatmap.py \
      --checkpoint ./runs/experiment_mixed_ddp/epoch_010.pt \
      --input small_testing/images \
      --output ./results_ddp_010/ \
      --vis \
      --eval

 python infer_cctag_heatmap.py --checkpoint ./runs/experiment_01/best.pt --input cctag_reallife.png --output ./results --vis
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 # 加速 generate_cctag_dataset.py（以你這組 640x400 / soft_focus_strength=0.9 / 3000 張 參數為目標）
 
   ## Summary
 
   目標是在不改變輸出格式與資料語意的前提下，加速你這條命令的生成速度：
 
   python generate_cctag_dataset.py \
     --num_images 3000 \
     --output_dir ./blur_dataset \
     --output_size 640x400 \
     --seed 42 \
     --marker_min 66 \
     --marker_max 333 \
     --partial_out_prob 0.13 \
     --partial_out_max_ratio 0.25 \
     --occ_min 0.85 \
     --occ_max 0.98 \
     --soft_focus_strength 0.9
 
   採用「盡量等價」策略：
 
   - 保留現有 CLI 與輸出目錄結構
   - 保留資料分布與標註語意
   - 只做低風險效能優化，不主動改資料內容規則
 
   ## Key Changes
 
   ### 1. 把背景生成改成全向量化
 
   在 composite_on_background() 移除 Python 的逐列 / 逐欄 gradient 迴圈，改成用 np.linspace 或 broadcasting 一次生成整張背景。
   這是低風險改動，輸出外觀等價，但能直接減少每張背景初始化成本。
 
   ### 2. 預先快取 heatmap 座標網格
 
   generate_gaussian_heatmap() 目前每張都重建 x/y/meshgrid。
   改成依 (heatmap_h, heatmap_w) 快取 xx/yy，每張只計算中心位移後的 Gaussian。
   你的設定固定 640x400 且 stride 固定，這個快取命中率會是 100%。
 
   ### 3. 降低 soft-focus 路徑中的重複大核 blur 成本
 
   apply_degradation() 在 soft_focus_strength=0.9 下是主要熱點，因為會做多次大 sigma GaussianBlur。
   優化方式：
 
   - 保留現有視覺流程
   - 只把可共用的中間結果整理清楚，避免不必要的 dtype 轉換與重複配置
   - 把 img.astype(np.float32)、np.clip(...).astype(np.uint8) 保持單進單出
   - 檢查能共用的模糊結果是否可直接衍生，避免額外臨時陣列
 
   不改 blur 次數與公式，先保結果穩定。
 
   ### 4. 降低標註幾何計算成本
 
   sample_circle_points() 現在每張都用 64 點取樣，再做 bbox / ellipse。
   對目前任務先做兩個低風險優化：
 
   - 預先快取 unit circle 的 cos/sin 樣本，避免每張重建角度表
   - bbox 與 ellipse 共用同一批 transformed points，避免任何重算
   - 若沒有 perspective transform，可直接走簡化分支，不必做 homogeneous transform
 
   ### 5. 把輸出寫檔改成較少 Python 開銷的批次方式
 
   目前每張都會：
 
   - cv2.imwrite
   - np.save
   - 開一個 YOLO txt
   - 寫一列 CSV
 
   保留輸出格式，但調整實作：
 
   - CSV 先累積到 list，最後一次 writer.writerows
   - YOLO 對空檔仍保留，但避免不必要的格式化重工
   - 視覺化 preview 只在 --visualize 時收集資料，維持現狀但把判斷留在最外層
   - 若有 --visualize，最多仍只保留前 16 張，不增加記憶體
 
   ### 6. 為三類配比模式加上「無重試」快速路徑
 
   目前 generate_single_sample() 在強制 boundary_target / normal_positive 時最多重試 64 次。
   保留行為，但加入更直接的生成策略：
 
   - boundary_target：直接用「保證中心出界」的取樣器，而不是靠反覆試到出界
   - normal_positive：直接用「保證中心在內」的取樣器，而不是 partial_out_prob=0 再走一般流程
   - 這會顯著降低三類比例模式下的額外隨機重試成本
 
   ### 7. 新增選擇性 multiprocessing，但放在第二階段
 
   先完成上面單進程優化。
   之後再新增可選 --num_workers：
 
   - 預設 1，保持相容
   - >1 時用 process pool 按 sample index 平行生成
   - 每個 worker 以 seed + index 派生獨立隨機狀態，保證可重現
   - 主程序只負責收集結果與寫檔
 
   這一項是第二階段，因為它改動最大，也最需要注意 deterministic behavior。
 
   ## Public Interface Changes
 
   ### 第一階段
 
   不新增必要 CLI，現有命令可直接受益。
 
   ### 第二階段
 
   新增可選：
 
   - --num_workers
       - 預設 1
       - 代表生成 worker 數量
       - 1 時行為與目前相同
 
   ## Test Plan
 
   1. 語法檢查
 
   - python -m py_compile generate_cctag_dataset.py visualize_random_labels.py
 
   2. 功能等價驗證
 
   - 用固定 seed 跑小資料集，確認仍正常輸出 images/, heatmaps/, labels_yolo/, labels.csv, config.json
   - 檢查正樣本、boundary target、empty negative 三類都仍能正確落標
 
   3. 視覺驗證
 
   - 對你這組模糊參數跑 --visualize
   - 確認 soft-focus 視覺特徵沒有明顯走樣
   - 確認 boundary target 還是落在邊界點
 
   4. 效能驗證
 
   - 用你這條命令做基準測試
   - 至少比較：
       - 優化前總耗時
       - 第一階段優化後總耗時
       - 若實作 --num_workers，再比較 1/2/4/8 workers
   - 主要觀察 img/s 與 CPU 使用率
 
   ## Assumptions
 
   - 目標是加速，不改輸出格式與資料語意
   - 目前以「低風險、盡量等價」為優先，不主動簡化 soft-focus 視覺模型
   - --num_workers 若實作，會被設計成完全可選，預設仍為單進程
   - 若第一階段單進程優化已足夠，就不必一定上 multiprocessing
