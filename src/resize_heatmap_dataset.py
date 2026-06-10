#!/usr/bin/env python3
"""Resize a heatmap-regression dataset to a new image resolution.

Resizes images, regenerates correctly-sized heatmaps, scales the pixel columns
in labels.csv (guarding -1 sentinels used by negatives), and copies the
resolution-independent YOLO labels verbatim. Filenames are preserved.

Example:
    uv run python src/resize_heatmap_dataset.py \
        --input_dir ./outputs/training_sets/real_world_merged_640x400 \
        --output_dir ./outputs/training_sets/real_world_merged_1024x640 \
        --width 1024 --height 640
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# labels.csv columns that hold absolute pixel values and must be scaled.
# x-scaled by sx, y-scaled by sy. Lengths (ellipse_a/_b) scale by sx (uniform).
X_COLS = {"x", "center_x", "ellipse_cx", "ellipse_a", "bbox_xmin", "bbox_xmax"}
Y_COLS = {"y", "center_y", "ellipse_cy", "ellipse_b", "bbox_ymin", "bbox_ymax"}
SENTINEL = -1.0  # value used for "no marker" rows; never scaled


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as d:
            return d["heatmap"]
    return np.load(path)


def save_heatmap(path: Path, hm: np.ndarray, src_suffix: str, dtype) -> None:
    hm = hm.astype(dtype)
    if src_suffix == ".npz":
        np.savez_compressed(path, heatmap=hm)
    else:
        np.save(path, hm)


def scale_csv_value(col: str, raw: str, sx: float, sy: float) -> str:
    try:
        val = float(raw)
    except ValueError:
        return raw
    if val == SENTINEL:
        return raw
    if col in X_COLS:
        return f"{val * sx:.6g}"
    if col in Y_COLS:
        return f"{val * sy:.6g}"
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--heatmap_stride", type=int, default=None,
                    help="Override stride; default reads config.json or falls back to 4")
    args = ap.parse_args()

    in_dir, out_dir = args.input_dir, args.output_dir
    if not in_dir.is_dir():
        raise SystemExit(f"input_dir not found: {in_dir}")

    # --- resolve source geometry / stride ---
    cfg_path = in_dir / "config.json"
    src_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    stride = args.heatmap_stride or src_cfg.get("heatmap_stride", 4)

    sample_img = next((in_dir / "images").glob("*"))
    in_h, in_w = cv2.imread(str(sample_img)).shape[:2]
    sx, sy = args.width / in_w, args.height / in_h
    if abs(sx - sy) > 1e-6:
        print(f"[warn] non-uniform scale sx={sx:.4f} sy={sy:.4f}; "
              f"ellipse axes scaled by sx only")

    new_hm_w, new_hm_h = args.width // stride, args.height // stride
    img_interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LANCZOS4
    print(f"resize {in_w}x{in_h} -> {args.width}x{args.height} "
          f"(scale {sx:.3f}x{sy:.3f}); heatmap -> {new_hm_w}x{new_hm_h}")

    for sub in ("images", "heatmaps", "labels_yolo"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- images ---
    imgs = sorted((in_dir / "images").glob("*"))
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        out = cv2.resize(im, (args.width, args.height), interpolation=img_interp)
        cv2.imwrite(str(out_dir / "images" / p.name), out)
        if (i + 1) % 500 == 0:
            print(f"  images {i + 1}/{len(imgs)}")
    print(f"  images done: {len(imgs)}")

    # --- heatmaps ---
    hms = sorted((in_dir / "heatmaps").glob("*"))
    for p in hms:
        hm = load_heatmap(p)
        out = cv2.resize(hm, (new_hm_w, new_hm_h), interpolation=cv2.INTER_LINEAR)
        save_heatmap(out_dir / "heatmaps" / p.name, out, p.suffix, hm.dtype)
    print(f"  heatmaps done: {len(hms)}")

    # --- yolo labels (normalized -> copy verbatim) ---
    yolos = sorted((in_dir / "labels_yolo").glob("*.txt"))
    for p in yolos:
        shutil.copy2(p, out_dir / "labels_yolo" / p.name)
    print(f"  yolo labels done: {len(yolos)}")

    # --- labels.csv (scale pixel columns) ---
    src_csv = in_dir / "labels.csv"
    if src_csv.exists():
        with src_csv.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        with (out_dir / "labels.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow([scale_csv_value(c, v, sx, sy)
                                 for c, v in zip(header, row)])
        print(f"  labels.csv done: {len(rows)} rows")

    # --- config.json ---
    new_cfg = dict(src_cfg)
    new_cfg.update({
        "output_size": [args.width, args.height],
        "output_width": args.width,
        "output_height": args.height,
        "heatmap_stride": stride,
        "heatmap_width": new_hm_w,
        "heatmap_height": new_hm_h,
        "heatmap_size": [new_hm_w, new_hm_h],
        "resized_from": f"{in_dir} ({in_w}x{in_h} -> {args.width}x{args.height})",
    })
    (out_dir / "config.json").write_text(json.dumps(new_cfg, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
