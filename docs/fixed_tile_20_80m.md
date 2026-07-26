# Fixed 1024x640 Tile: 20-80 m Dataset Workflow

The no-occlusion source set excludes `6f_labeled_1024x640_roi_occ` and applies
the session holdout/buffer exclusions. Its positive diameter counts from
`2 * max(ellipse_a, ellipse_b)` are:

| Diameter | Available | Materialized target | Missing |
| --- | ---: | ---: | ---: |
| 48-80 px | 145 | 4,500 | 4,355 |
| 80-128 px | 371 | 4,500 | 4,129 |
| 128-192 px | 1,255 | 3,000 | 1,745 |
| 192-320 px | 2,375 | 3,000 | 625 |
| Negatives | 9,868 | 10,000 | 132 |

`>320 px` positives are intentionally excluded. Validation sources must be
split by capture session before this workflow; do not pass validation sessions
to the materializer.

## 1. Generate the missing small positives

```bash
OUTPUT_ROOT=/mnt/tmp1/r13922171/fixed_tile_20_80m_supplement_no_occ_v2 \
scripts/generate_fixed_tile_small_positives.sh
```

Each diameter bin is 65% clean, 25% image-degraded, and 10% tile-boundary.
All modes have physical occlusion disabled. Clean/degraded rows must be fully
visible and unclamped; boundary rows must be 25-99.9% visible with a clamped
nearest-boundary target. Every positive is accepted using its final label
diameter, visibility, and clamp state. The old `*_supplement_no_occ` output is
a legacy candidate and must not be materialized.

## 2. Materialize the train-only dataset

Pass the existing train-only sources and all supplement shard directories:

```bash
supplement_sources=()
for path in /mnt/tmp1/r13922171/fixed_tile_20_80m_supplement_no_occ_v2/*; do
  supplement_sources+=(--source "$path")
done

uv run python scripts/build_fixed_tile_dataset.py \
  --source outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
  --source outputs/datasets/6f_labeled_1024x640_roi \
  --source outputs/training_sets/real_world_merged_1024x640 \
  --source outputs/datasets/hard_negative_random_20260608_230541_1024x640 \
  --source outputs/datasets/hard_negative_random_20260608_231324_1024x640 \
  --source outputs/datasets/hard_negative_random_20260608_231516_1024x640 \
  "${supplement_sources[@]}" \
  --exclude-list data_diagnosis/split_lists/real_holdout.txt \
  --exclude-list data_diagnosis/split_lists/real_buffer_dropped.txt \
  --output /mnt/tmp1/r13922171/fixed_tile_20_80m_train_no_occ \
  --positive-target 15000 \
  --negative-target 10000
```

The materializer prints a per-source/per-scale preflight audit and rejects
occluded, low-visibility, and wrong-clamp positives before sampling. Artifacts are hard-linked by default to avoid duplicating the existing data.
Treat source and output artifacts as immutable. Use `--copy-files` only when an
independent physical copy is required. The repository path
Then link `outputs/training_sets/fixed_tile_20_80m_train_no_occ` to this output.

## 3. Fine-tune

```bash
scripts/run_fixed_tile_20_80m_no_occ.sh
```

The launcher runs a fail-closed dataset gate before loading `epoch_070.pt`.

The run starts from `fable_occ_hrnet_w18/epoch_070.pt`, but creates a fresh
AdamW optimizer at `1e-4`; it does not resume epoch 71 or retain its optimizer
state. It keeps HRNet-W18 at 1024x640, focal loss, offset and size heads, and
trains for 25 epochs. A micro-batch of five gives exactly
three positives and two negatives per rank. Positive scale quotas converge to
30/30/20/20 across each epoch.

Validation `metrics.csv` includes recall, mean peak, and detected-positive
center error for each scale bin, plus overall negative false-positive rate and
tile-boundary recall.
