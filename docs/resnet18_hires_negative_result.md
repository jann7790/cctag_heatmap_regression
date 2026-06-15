# resnet18_hires backbone — negative result (abandoned)

Record of a backbone experiment that **did not work**. The `resnet18_hires`
option still exists in the code, but it is a dead end: **do not select it**.
Plain `resnet18` remains the default/production backbone.

Branch: `resnet18-hires-stem-decoder` (merged to `main`).

## What was tried

A new backbone `--backbone resnet18_hires` (class `CCTagNetResNet18HiRes`, mirrored
in both `src/train_cctag_heatmap_ddp.py` and `src/infer_cctag_heatmap.py`). Goal:
lower `center_l2_px` for sub-pixel center localization. Two changes vs `resnet18`,
made together on purpose:

- **Stem**: drop the ResNet `maxpool` (keep `conv1` stride 2) → encoder pyramid 2×
  finer, bottleneck stride 32 → 16, so the peak location is not destroyed early.
- **Decoder**: double-conv blocks (2× 3×3 per stage), final feature 64ch (was 32),
  3 up-stages instead of 4. Output heatmap stays **256×160** (unchanged).

Why both at once: an earlier experiment showed that raising the *output* resolution
alone lost everywhere ("don't raise resolution without decoder capacity"). This
variant keeps the 256×160 output and instead gives a finer encoder + beefier
decoder, isolating it from that earlier failure.

## Result

A/B baseline = `outputs/runs/heatmap_1024` (plain resnet18, input 1024×640,
offset+size+focal, same `generated_training_sets_1024/mixed`).

Run `outputs/runs/resnet18_hires` (batch 12, 1024×640, same data/val):

| metric | resnet18_hires | baseline (plain resnet18) |
|---|---|---|
| val_loss / detection | healthy (~0.99), FPR 0.83 → 0.02 | healthy |
| `center_l2_px` | **202px @ ep1 → 108px @ ep12** | **27px @ ep7** |

Localization was ~4× worse at matching epochs and never recovered. Tell-tale:
val_loss (heatmap-only) was good while `center_l2` stayed awful → the problem is in
the decode path / peak placement, not the heatmap quality.

**Hypothesis (held up):** dropping `maxpool` puts the pretrained ResNet layers at a
feature scale they were not trained for → detection recovers fast but precise
localization re-fits very slowly.

## Verdict

**Abandoned.** Do not revisit the drop-maxpool ResNet hack. The residual error on the
production model is edge + small-marker discretization, **not** the backbone, so this
whole "finer stem" direction was the wrong lever. If a gentler stem is ever wanted
again, prefer `efficientnet_b0` (natively gentle stem, no pretraining mismatch) over
hacking ResNet's `maxpool`.

`resnet18_hires` is kept only as an unused `--backbone` choice; checkpoints are not
interchangeable with `resnet18`.

## Useful work that merged along on the same branch

The branch also carried changes that **are** kept in use:

- `--occ_loss_weight`: per-sample heatmap-loss weighting by occlusion ratio
  (`weight = 1 + k*occ_ratio`, mean-normalized) to focus on hard-occluded markers.
- `channels_last` now default-on; new `--tf32` toggle; `--save_every` default 1 → 10.
- `generate_cctag_dataset.py`: `apply_hardware_occlusion` (rotated-rect / ellipse
  hardware-style occluders).
- `augment_roi_occlusion.py`: dropped the bespoke cable/bezier occluder in favour of
  the shared hardware-occlusion path; tightened occlusion-ratio defaults.
- `eval_l2_distribution.py`: mean-L2-by-occlusion-bin reporting.
- New tooling: `bench_deploy_strategies.py`, `eval_negative_peaks.py`,
  `restride_heatmap_dataset.py`, `make_dataset_samples_grid.py`,
  `make_heatmap_triptych.py`.
