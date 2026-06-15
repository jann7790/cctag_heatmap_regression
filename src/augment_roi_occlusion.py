#!/usr/bin/env python3
"""Augment a ROI heatmap dataset with synthetic occlusion over real markers.

Takes a labeled ROI dataset (e.g. ``outputs/datasets/6f_labeled_1024x640_roi``
produced by ``src/sample_roi_dataset.py``) and writes a NEW dataset that adds
occluded copies of every positive sample, reusing the same occluder library as
the synthetic generator (``apply_random_occlusion`` in
``src/generate_cctag_dataset.py``).

Occlusion is drawn over the real marker; it does NOT move the center, so the
center label and the existing heatmap stay valid (the model keeps learning to
regress the center under occlusion). Only ``occlusion_ratio`` is updated to the
measured coverage. Negatives are never occluded.

The default ``--occlusion_style hardware`` paints a realistic FSO-terminal
occluder (mount bracket, shiny collimator barrel, clamp arm, support post, and
blocks sitting on the outer rings) centered on the marker, matching the real rig
(see ``capture_20260608_*.png``). ``apply_hardware_occlusion`` already renders
gradient shading, specular highlights, a soft drop shadow and feathered edges,
so no extra post-processing is applied.

Tiers control the occluder scale: variants alternate between a low/partial range
and a hard range, so half the occluded copies are partial and half are heavy.

The source dataset is never modified; the output goes to a fresh directory and
is drop-in compatible with ``src/train_cctag_heatmap_ddp.py`` (images/,
heatmaps/ NPZ, labels_yolo/, labels.csv with 23 columns, config.json).

Defaults: occluded positives only (``--no-keep_clean``), and markers whose
ellipse extends beyond the ROI frame are skipped (``--skip_out_of_frame``).
Pass ``--keep_clean`` to also copy the clean originals + negatives, or
``--no-skip_out_of_frame`` to occlude cut-off markers too.

Example (canonical: occluded positives only, in-frame markers only):
    uv run python src/augment_roi_occlusion.py \
        --input_dir ./outputs/datasets/6f_labeled_1024x640_roi \
        --output_dir ./outputs/datasets/6f_labeled_1024x640_roi_occ \
        --variants_per_positive 2 --no-keep_clean --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the synthetic occluder library. generate_cctag_dataset.py only defines
# constants at module scope (main is guarded by __name__), so the import is
# side-effect free. Adding the script's own directory keeps the import working
# regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_cctag_dataset import apply_random_occlusion  # noqa: E402

# 23-column schema, identical order to src/sample_roi_dataset.py:38-46.
CSV_HEADER = [
    "filename", "x", "y", "center_x", "center_y",
    "ellipse_cx", "ellipse_cy", "ellipse_a", "ellipse_b", "ellipse_angle_rad",
    "occlusion_ratio",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    "yolo_cx", "yolo_cy", "yolo_w", "yolo_h",
    "is_negative", "negative_mode", "has_visible_marker", "visible_marker_ratio",
    "target_clamped",
]


def ensure_absent(target: Path) -> None:
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing path: {target}")


def marker_radius_from_row(row: dict, scale: float) -> float:
    """Occlusion drawing radius from the fitted ellipse.

    ``ellipse_a/b`` are the outer semi-axes (the bbox is ~2*ellipse_a wide).
    The synthetic generator passes the inner radius (~half the outer radius) to
    apply_random_occlusion, so we mirror that with (a+b)/4 to keep occluder
    sizing and the measured occlusion_ratio consistent with the trained-on
    synthetic data."""
    a = float(row.get("ellipse_a") or 0.0)
    b = float(row.get("ellipse_b") or 0.0)
    if a > 0.0 and b > 0.0:
        return max((a + b) / 4.0 * scale, 8.0)
    # Fallback: derive from the bbox if the ellipse fit is missing.
    bw = float(row.get("bbox_xmax") or 0.0) - float(row.get("bbox_xmin") or 0.0)
    bh = float(row.get("bbox_ymax") or 0.0) - float(row.get("bbox_ymin") or 0.0)
    return max((bw + bh) / 8.0 * scale, 8.0)


def marker_exceeds_frame(row: dict, frame_w: int, frame_h: int) -> bool:
    """True if any part of the marker ellipse extends beyond the ROI frame.

    Uses the axis-aligned half-extents of the rotated ellipse (semi-axes
    ellipse_a/b, angle ellipse_angle_rad). Such markers are already cut off by
    the image boundary, so we don't add occlusion on top of them."""
    cx = float(row.get("center_x") or row.get("x") or 0.0)
    cy = float(row.get("center_y") or row.get("y") or 0.0)
    a = float(row.get("ellipse_a") or 0.0)
    b = float(row.get("ellipse_b") or 0.0)
    th = float(row.get("ellipse_angle_rad") or 0.0)
    ext_x = math.hypot(a * math.cos(th), b * math.sin(th))
    ext_y = math.hypot(a * math.sin(th), b * math.cos(th))
    return (cx - ext_x < 0.0 or cx + ext_x > frame_w
            or cy - ext_y < 0.0 or cy + ext_y > frame_h)


def is_positive(row: dict) -> bool:
    if int(float(row.get("is_negative", "1") or 1)) != 0:
        return False
    cx = float(row.get("center_x") or row.get("x") or -1.0)
    cy = float(row.get("center_y") or row.get("y") or -1.0)
    return np.isfinite(cx) and np.isfinite(cy) and cx >= 0.0 and cy >= 0.0


def copy_clean(stem: str, in_dir: Path, out_dir: Path) -> None:
    """Copy an existing sample's image / heatmap / yolo files verbatim."""
    shutil.copyfile(in_dir / "images" / f"{stem}.png", out_dir / "images" / f"{stem}.png")
    shutil.copyfile(in_dir / "heatmaps" / f"{stem}.npz", out_dir / "heatmaps" / f"{stem}.npz")
    src_txt = in_dir / "labels_yolo" / f"{stem}.txt"
    dst_txt = out_dir / "labels_yolo" / f"{stem}.txt"
    if src_txt.is_file():
        shutil.copyfile(src_txt, dst_txt)
    else:
        dst_txt.write_text("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi_occ_v2"))
    ap.add_argument("--variants_per_positive", type=int, default=2,
                    help="Occluded copies generated per positive sample.")
    ap.add_argument("--keep_clean", action=argparse.BooleanOptionalAction, default=False,
                    help="Also copy the clean originals (positives + negatives) into the output "
                         "so the dataset is self-contained. Default off (occluded positives only).")
    ap.add_argument("--occ_low_min", type=float, default=0.05)
    ap.add_argument("--occ_low_max", type=float, default=0.45)
    ap.add_argument("--occ_hard_min", type=float, default=0.45)
    ap.add_argument("--occ_hard_max", type=float, default=0.65)
    ap.add_argument("--max_occ_ratio", type=float, default=0.75,
                    help="Hard cap on measured marker coverage. The marker must keep at least "
                         "(1 - max_occ_ratio) of its outer disc visible, so the center stays "
                         "localizable; occluders exceeding this are redrawn (see --occ_max_attempts).")
    ap.add_argument("--occ_max_attempts", type=int, default=5,
                    help="Max redraws to satisfy --max_occ_ratio; the least-covering attempt is "
                         "kept if none qualify.")
    ap.add_argument("--occlusion_style", default="hardware",
                    choices=["standard", "aggressive", "center_heavy",
                             "hardware", "mixed"],
                    help="Occluder style passed to apply_random_occlusion. 'hardware' paints a "
                         "realistic FSO-terminal occluder (bracket/barrel/clamp/post + outer-ring "
                         "blocks) matching the real rig; 'mixed' uses hardware 60%% of the time.")
    ap.add_argument("--occluder_templates", default="auto")
    ap.add_argument("--occ_radius_scale", type=float, default=1.0,
                    help="Multiplier on the (a+b)/4 occluder radius.")
    ap.add_argument("--skip_out_of_frame", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip occluding markers whose ellipse extends beyond the ROI frame "
                         "(they are already cut off by the boundary). Use --no-skip_out_of_frame "
                         "to occlude them anyway.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.variants_per_positive < 1:
        raise SystemExit("--variants_per_positive must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    in_dir = args.input_dir
    src_csv = in_dir / "labels.csv"
    if not src_csv.is_file():
        raise FileNotFoundError(src_csv)

    out_dir = args.output_dir
    ensure_absent(out_dir)
    img_dir, hm_dir, yolo_dir = out_dir / "images", out_dir / "heatmaps", out_dir / "labels_yolo"
    for d in (img_dir, hm_dir, yolo_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(src_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        src_rows = list(reader)
        header = reader.fieldnames or CSV_HEADER

    # Frame size for the out-of-frame test: prefer the source config, else the
    # first image's dimensions.
    frame_w = frame_h = None
    cfg_path = in_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        frame_w, frame_h = cfg.get("image_width"), cfg.get("image_height")
    if not frame_w or not frame_h:
        probe = cv2.imread(str(in_dir / "images" / f"{src_rows[0]['filename']}.png"))
        frame_h, frame_w = probe.shape[:2]
    frame_w, frame_h = int(frame_w), int(frame_h)

    tiers = [
        (args.occ_low_min, args.occ_low_max),
        (args.occ_hard_min, args.occ_hard_max),
    ]

    out_rows: list[dict] = []
    n_clean = n_occ = n_pos = n_oof = 0

    for row in src_rows:
        stem = row["filename"]

        if args.keep_clean:
            copy_clean(stem, in_dir, out_dir)
            out_rows.append(dict(row))
            n_clean += 1

        if not is_positive(row):
            continue
        n_pos += 1

        if args.skip_out_of_frame and marker_exceeds_frame(row, frame_w, frame_h):
            n_oof += 1
            continue

        img_path = in_dir / "images" / f"{stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] cannot read {img_path}, skipping occlusion for it")
            continue

        cx = float(row.get("center_x") or row.get("x"))
        cy = float(row.get("center_y") or row.get("y"))
        radius = marker_radius_from_row(row, args.occ_radius_scale)

        for v in range(args.variants_per_positive):
            lo, hi = tiers[v % len(tiers)]
            base = img.copy()
            # Redraw until the marker keeps enough visible area to localize the
            # center; a fully-occluded marker is label noise for the heatmap.
            best = None  # (ratio, occluded)
            for _ in range(max(1, args.occ_max_attempts)):
                occluded, actual_ratio = apply_random_occlusion(
                    base.copy(), (cx, cy), radius,
                    occlusion_range=(lo, hi),
                    occlusion_style=args.occlusion_style,
                    occ_distribution="uniform",
                    occluder_templates=args.occluder_templates,
                )
                if best is None or actual_ratio < best[0]:
                    best = (actual_ratio, occluded)
                if actual_ratio <= args.max_occ_ratio:
                    break
            actual_ratio, occluded = best

            out_stem = f"{stem}_occ{v}"
            cv2.imwrite(str(img_dir / f"{out_stem}.png"), occluded)
            # Center is unchanged -> reuse the source heatmap and yolo bbox.
            shutil.copyfile(in_dir / "heatmaps" / f"{stem}.npz", hm_dir / f"{out_stem}.npz")
            src_txt = in_dir / "labels_yolo" / f"{stem}.txt"
            (yolo_dir / f"{out_stem}.txt").write_text(
                src_txt.read_text() if src_txt.is_file() else ""
            )

            new_row = dict(row)
            new_row["filename"] = out_stem
            new_row["occlusion_ratio"] = f"{actual_ratio:.4f}"
            out_rows.append(new_row)
            n_occ += 1

    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)

    config = {
        "generation_type": "cctag_roi_occlusion_augmented",
        "source_dataset": str(in_dir),
        "keep_clean": bool(args.keep_clean),
        "variants_per_positive": args.variants_per_positive,
        "occlusion_style": args.occlusion_style,
        "occluder_templates": args.occluder_templates,
        "occ_radius_scale": args.occ_radius_scale,
        "max_occ_ratio": args.max_occ_ratio,
        "occ_max_attempts": args.occ_max_attempts,
        "skip_out_of_frame": bool(args.skip_out_of_frame),
        "frame_size": [frame_w, frame_h],
        "tiers": {"low": [args.occ_low_min, args.occ_low_max],
                  "hard": [args.occ_hard_min, args.occ_hard_max]},
        "seed": args.seed,
        "num_source_rows": len(src_rows),
        "num_source_positives": n_pos,
        "num_skipped_out_of_frame": n_oof,
        "num_occluded_positives": n_pos - n_oof,
        "num_clean_copied": n_clean,
        "num_occluded": n_occ,
        "num_samples": len(out_rows),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Done. {n_pos} positives ({n_oof} out-of-frame skipped) -> {n_occ} occluded copies"
          + (f" + {n_clean} clean originals" if args.keep_clean else "")
          + f" = {len(out_rows)} samples")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
