# Real-data ROI dataset + occlusion augmentation

How we turn the real labeled captures (`6f_labeled`) into a heatmap training set
and then add synthetic occlusion on top of the real markers. Two standalone
scripts, run in order.

```
6f_labeled (4096x2160 frames)
   │  src/sample_roi_dataset.py     (crop rotated 1024x640 ROIs, no resize)
   ▼
6f_labeled_1024x640_roi            (clean positives + negatives)
   │  src/augment_roi_occlusion.py  (draw occluders over real markers)
   ▼
6f_labeled_1024x640_roi_occ        (occluded positives)
```

Both outputs are drop-in compatible with `src/train_cctag_heatmap_ddp.py`
(`images/`, `heatmaps/` NPZ float16, `labels_yolo/`, `labels.csv`, `config.json`)
and are meant to be **mixed with the synthetic 1024 set** during training.

## Stage 1 — `sample_roi_dataset.py`

For each marker, crops `pos_per_marker` (default 4) rotated `1024x640` ROIs that
place the marker at a random position with a random rotation (±15°) via a single
affine `M=[R(θ)|b]`; labels transform with the same `M` (ellipse_a/b unchanged,
angle `+θ`). **No scaling** — native pixels are preserved so `ellipse_a/b` stay
valid and the crop matches the no-resize deployment tiling. Negatives are random
rotated ROIs whose crop excludes the marker center. Heatmaps regenerated at
stride 4 → 256x160, sigma 3.0.

> Note: close-range frames produce markers larger than the tile. The current set
> had positives with diameter (`2*max(ellipse_a,b)`) > 900 px removed post-hoc
> (`config.json: filtered_max_marker_diam_px: 900`).

## Stage 2 — `augment_roi_occlusion.py`

Adds occluded copies of every positive, **reusing the synthetic occluder
library** — it imports `apply_random_occlusion` from `generate_cctag_dataset.py`
for the base geometric block (single / scatter / tshape / cross / hshape / …
templates), then (by default, `--realistic`) **dresses it up to match the real
rig** (see `capture_20260608_*.png`): curved cables crossing the marker,
bright metallic glints on the occluder, and feathered / motion-blurred edges.
`occlusion_ratio` is recomputed from the final soft coverage. Pass
`--no-realistic` for the legacy flat-dark-block behaviour.

Canonical run (defaults already match this):

```bash
uv run python src/augment_roi_occlusion.py \
  --input_dir ./outputs/datasets/6f_labeled_1024x640_roi \
  --output_dir ./outputs/datasets/6f_labeled_1024x640_roi_occ \
  --variants_per_positive 2 --seed 42
```

### Design decisions (the non-obvious bits)

- **Occlusion never moves the center.** So the existing heatmap and YOLO bbox are
  copied verbatim and only `occlusion_ratio` is recomputed to the measured
  coverage. Heavily-occluded copies stay positive (peak intact) — the model
  learns to regress the center *under* occlusion. The trainer decides
  positive/negative from the heatmap peak (`gt_peak > 0.1`), not from
  `occlusion_ratio`.

- **Occluder radius = `(ellipse_a + ellipse_b) / 4`.** `ellipse_a/b` are the
  *outer* semi-axes (bbox ≈ `2*ellipse_a` wide). The synthetic generator passes
  the *inner* radius (~half the outer radius) to `apply_random_occlusion`, so we
  mirror that with `/4` to keep occluder sizing and the measured
  `occlusion_ratio` consistent with the trained-on synthetic data.

- **Mixed low+hard tiers**, mirroring `scripts/generate_training_sets.sh`:
  variants alternate between `0.05–0.50` (partial) and `0.50–0.85` (heavy), with
  `occlusion_style aggressive` for *both* — the shell script uses aggressive even
  for `base_low_occ`, so the tier only changes occluder *complexity*, not style.

- **Out-of-frame markers are skipped** (`--skip_out_of_frame`, default on). A
  marker whose ellipse extends past the ROI boundary (computed from the rotated
  ellipse's axis-aligned half-extents) is already cut off; we don't stack
  occlusion on top of it. On the current set this drops 1194 / 2936 positives.

- **Source is never modified.** Output goes to a fresh dir; the script refuses to
  overwrite an existing path.

### Defaults & flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--keep_clean` | off | also copy clean originals (pos + neg) into output; off ⇒ occluded positives only |
| `--skip_out_of_frame` | on | skip markers cut off by the frame edge |
| `--variants_per_positive` | 2 | occluded copies per in-frame positive |
| `--occ_low_min/max` | 0.05 / 0.50 | partial tier range |
| `--occ_hard_min/max` | 0.50 / 0.85 | heavy tier range |
| `--occ_radius_scale` | 1.0 | multiplier on the `(a+b)/4` radius |
| `--realistic` | on | add cables + metallic glints + soft/motion-blurred edges over the block |
| `--cable_prob` | 0.6 | chance of drawing 1–2 curved cables across the marker |
| `--metallic_prob` | 0.5 | chance of adding bright metallic-glint streaks on the occluder |
| `--edge_blur` | 5 | gaussian kernel (px) feathering the occluder edge (0 = off) |
| `--motion_blur_max` | 9 | max directional motion-blur kernel (px); per-variant random in `[0,max]`, 0 = off |

With the defaults on the current set: `2936 positives (1194 out-of-frame
skipped) → 1742 × 2 = 3484 occluded positives`.

## Training

Joint-train with the synthetic 1024 set and this ROI set (MUST pass
`--input_width 1024 --input_height 640`):

```bash
--train_dataset_dir .../generated_training_sets_1024/mixed_train_dataset \
--train_dataset_dir outputs/datasets/6f_labeled_1024x640_roi_occ
```

## QC

```bash
uv run python src/visualize_random_labels.py \
  --dataset_dir ./outputs/datasets/6f_labeled_1024x640_roi_occ \
  --num_samples 9 --show_yolo_bbox --output ./outputs/tmp/roi_occ_labels.jpg
```

Confirm the occluder sits on the marker and the center label/heatmap peak is
still on the real center.
