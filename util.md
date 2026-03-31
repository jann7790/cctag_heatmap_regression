• 新增了 visualize_random_heatmaps.py，會把 heatmaps/*.npy 疊到原圖上，並標出：

  - 白色十字：labels.csv 的中心點
  - 黃色斜十字：heatmap peak
  - 左上角文字：檔名、heatmap 最大值、occlusion ratio

  我已經直接幫你跑了一張檢查圖，輸出在 tmp/cctag_dataset_640_400_heatmap_overlay.jpg。

  之後你自己可用這個指令重跑：

  python visualize_random_heatmaps.py \
    --dataset_dir ./cctag_dataset_640_400 \
    --num_samples 9 \
    --seed 42 \
    --output ./tmp/cctag_dataset_640_400_heatmap_overlay.jpg


python train_cctag_heatmap.py \
    --dataset_dir ./blur_dataset \
    --output_dir ./runs/experiment_01 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --train_ratio 0.9 \
    --num_workers 4 \
    --seed 42 \
    --input_width 640 \
    --input_height 400 \
    --save_every 5 \
    --device cuda


uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple
