
  # Repo + 本機工作目錄整理方案

  ## Summary

  這次整理不只重整 repo 結構，也把本機產物收斂到單一 outputs/ 根目錄，並同步調整 .gitignore。目標是讓根目錄只留下原始碼、文件、資產與腳本；dataset、training runs、inference
  results、tmp 類內容全部集中管理。

  預設決策：

  - 本機輸出統一集中到 outputs/
  - 既有歷史輸出資料夾納入搬遷規劃
  - .gitignore 採保守兼容：保留舊規則，同時加入新結構規則

  ## Key Changes

  ### 1. 目標結構

  整理後的 repo / 本機結構：

  .
  ├── src/
  ├── scripts/
  ├── docs/
  ├── assets/
  │   ├── markers/
  │   └── samples/
  ├── outputs/
  │   ├── datasets/
  │   ├── testing/
  │   ├── training_sets/
  │   ├── runs/
  │   ├── inference/
  │   └── tmp/
  ├── README.md
  ├── requirements.txt
  └── .gitignore

  規則：

  - 正式 source、docs、scripts、assets 留在 repo 根目錄可見位置。
  - 所有本機生成內容只允許落在 outputs/ 下。
  - 根目錄不再保留 15000_dataset/、small_testing/、generated_training_sets/、results_ddp/ 這類工作產物。

  ### 2. 歷史輸出搬遷對照

  把目前根目錄既有資料夾整理到新位置：

  - 15000_dataset/ -> outputs/datasets/15000_dataset/
  - 2tmp/ -> outputs/tmp/2tmp/
  - cctag_dataset_640_400/ -> outputs/datasets/cctag_dataset_640_400/
  - generated_training_sets/ -> outputs/training_sets/generated_training_sets/
  - generated_testing_suites/ -> outputs/testing/generated_testing_suites/
  - runs/ -> outputs/runs/
  - results_ddp/ -> outputs/inference/results_ddp/
  - results_ddp_010/ -> outputs/inference/results_ddp_010/
  - small_testing/ -> outputs/testing/small_testing/
  - testing_small_hard/ -> outputs/testing/testing_small_hard/
  - testing_extreme_mix/ -> outputs/testing/testing_extreme_mix/
  - ultimate_dataset/ -> outputs/datasets/ultimate_dataset/

  額外規則：

  - broken.png、covered.png 視為暫時產物，移到 outputs/tmp/ 或直接不保留。
  - cctag.png、cctag0.png、cctag_reallife.png 若要保留示例用途，移到 assets/samples/。

  ### 3. .gitignore 調整策略

  .gitignore 改成「新規則優先、舊規則兼容」：

  保留並整理這幾段：

  - Python / editor / local env 規則
  - 本地工具狀態，例如 .claude/
  - 新主規則：outputs/
  - 過渡兼容：保留現有 runs/、*_dataset/、generated_*、testing_* 等舊規則，避免搬遷前後漏掉
  - 暫時圖片產物規則：broken.png、covered.png

  建議方向：

  - README 與 shell scripts 之後只示範 outputs/... 路徑
  - 舊 ignore pattern 先保留 1 個整理週期，等所有工作流都完成切換後再刪除舊特例

  ### 4. 腳本與文件同步

  所有入口改成以 outputs/ 為預設輸出根：

  - generator 預設示例輸出到 outputs/datasets/...
  - training scripts 預設 checkpoint 到 outputs/runs/...
  - inference scripts 預設結果到 outputs/inference/...
  - shell workflow 預設：
      - generate_training_sets.sh -> outputs/training_sets/...
      - generate_testing_suites.sh -> outputs/testing/...
      - generate_and_train_mixed.sh -> 讀寫 outputs/training_sets/... 與 outputs/runs/...

  README / docs 同步更新：

  - 明確寫出「repo 內容」與「本機產物」邊界
  - 提供 3 個標準路徑範例：dataset、run、inference output
  - 說明舊根目錄資料夾是歷史產物，未來不再使用

  ### 5. 搬遷執行順序

  實作時按這個順序做，避免混亂：

  1. 先更新 .gitignore
  2. 建立 outputs/ 目錄結構與命名慣例
  3. 更新腳本與 README 的預設輸出路徑
  4. 搬移歷史本機輸出資料夾到 outputs/
  5. 搬移示例圖片到 assets/samples/
  6. 清掉根目錄遺留的暫時產物與空資料夾
  7. 驗證所有常用指令仍能跑通

  ## Test Plan

  1. git status --ignored 應顯示 outputs/ 下內容被統一忽略，根目錄不再散落大量 ignored outputs。
  2. 跑一次小型 generator，確認輸出落在 outputs/datasets/...。
  3. 跑一次 visualization，確認能從新資料集路徑讀檔。
  4. 跑一次 train smoke test，確認 checkpoint 落在 outputs/runs/...。
  5. 跑一次 infer，確認結果落在 outputs/inference/...。
  6. 檢查 README 中所有示例命令，不再引用舊根目錄輸出路徑。
  7. 檢查 git ls-files，確保沒有新的大型本機產物被誤納入版本控制。

  ## Assumptions

  - 你要的是「本機工作目錄也變乾淨」，不是只做 repo 內部重構。
  - 歷史輸出資料仍可保留，但要搬到 outputs/ 下統一管理。
  - .gitignore 現階段以兼容為主，不立即刪掉所有舊規則。
  - 根目錄只保留正式內容；示例圖片若有展示價值才保留到 assets/samples/。
  - readme.md 目前有未提交修改，實作時必須整合內容，不可直接覆蓋。
