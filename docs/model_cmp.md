  ---
  Comparison: experiment_mixed_ddp vs experiment_mixed_v3

  testing_boundary_hard (850 positive / 150 negative)

  ┌─────────────────────┬────────┬────────┐
  │       Metric        │  DDP   │   V3   │
  ├─────────────────────┼────────┼────────┤
  │ detection_accuracy  │ 0.9950 │ 0.8500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_precision │ 1.0000 │ 0.8500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_recall    │ 0.9941 │ 1.0000 │
  ├─────────────────────┼────────┼────────┤
  │ detection_f1        │ 0.9971 │ 0.9189 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_f1    │ 0.7429 │ 0.7151 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_iou   │ 0.5909 │ 0.5566 │
  ├─────────────────────┼────────┼────────┤
  │ mean_center_l2_px   │ 6.41   │ 9.68   │
  └─────────────────────┴────────┴────────┘

  ---
  testing_extreme_mix (750 positive / 250 negative)

  ┌─────────────────────┬────────┬────────┐
  │       Metric        │  DDP   │   V3   │
  ├─────────────────────┼────────┼────────┤
  │ detection_accuracy  │ 0.9790 │ 0.7500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_precision │ 1.0000 │ 0.7500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_recall    │ 0.9720 │ 1.0000 │
  ├─────────────────────┼────────┼────────┤
  │ detection_f1        │ 0.9858 │ 0.8571 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_f1    │ 0.7482 │ 0.6997 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_iou   │ 0.5977 │ 0.5381 │
  ├─────────────────────┼────────┼────────┤
  │ mean_center_l2_px   │ 5.91   │ 15.65  │
  └─────────────────────┴────────┴────────┘

  ---
  testing_negative_hard (650 positive / 350 negative)

  ┌─────────────────────┬────────┬────────┐
  │       Metric        │  DDP   │   V3   │
  ├─────────────────────┼────────┼────────┤
  │ detection_accuracy  │ 1.0000 │ 0.6500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_precision │ 1.0000 │ 0.6500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_recall    │ 1.0000 │ 1.0000 │
  ├─────────────────────┼────────┼────────┤
  │ detection_f1        │ 1.0000 │ 0.7879 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_f1    │ 0.7500 │ 0.6872 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_iou   │ 0.6000 │ 0.5234 │
  ├─────────────────────┼────────┼────────┤
  │ mean_center_l2_px   │ 5.21   │ 4.91   │
  └─────────────────────┴────────┴────────┘

  ---
  testing_small_hard (850 positive / 150 negative)

  ┌─────────────────────┬────────┬────────┐
  │       Metric        │  DDP   │   V3   │
  ├─────────────────────┼────────┼────────┤
  │ detection_accuracy  │ 0.9860 │ 0.8500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_precision │ 1.0000 │ 0.8500 │
  ├─────────────────────┼────────┼────────┤
  │ detection_recall    │ 0.9835 │ 1.0000 │
  ├─────────────────────┼────────┼────────┤
  │ detection_f1        │ 0.9917 │ 0.9189 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_f1    │ 0.7535 │ 0.7220 │
  ├─────────────────────┼────────┼────────┤
  │ heatmap_pixel_iou   │ 0.6045 │ 0.5650 │
  ├─────────────────────┼────────┼────────┤
  │ mean_center_l2_px   │ 4.57   │ 13.79  │
  └─────────────────────┴────────┴────────┘

  ---
  Summary

  DDP wins decisively across all 4 suites. The V3 model has a critical weakness: its detection precision is exactly
   equal to the positive ratio in each suite (0.85, 0.75, 0.65, 0.85), meaning it predicts every image as positive
  (recall = 1.000 everywhere). It never outputs "no detection," making it useless for negative rejection. DDP
  correctly rejects negatives (precision = 1.000) while maintaining high recall.
