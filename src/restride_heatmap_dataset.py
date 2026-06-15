#!/usr/bin/env python3
"""Re-render a heatmap dataset's GT heatmaps at a finer stride (higher-res, sharper).

Unlike ``resize_heatmap_dataset.py`` -- which upsamples the existing coarse
heatmap with cv2.resize and therefore cannot recover sub-pixel sharpness -- this
RE-RENDERS each Gaussian directly from the ``labels.csv`` center at the new
resolution. The image resolution is unchanged; only ``heatmaps/`` and the heatmap
fields of ``config.json`` change. Images, YOLO labels and labels.csv are copied
verbatim (centers are already in source-image pixels, so they stay valid).

Negatives (``is_negative=1``) get all-zero heatmaps. The per-dataset Gaussian
sigma is auto-estimated from the existing heatmaps so the physical peak width is
preserved across the stride change (sigma in cells scales with the resolution).

Example (stride 4 -> 2, i.e. 256x160 -> 512x320 for 1024x640 images):
  uv run python src/restride_heatmap_dataset.py \
    --input_dir outputs/datasets/6f_labeled_1024x640_roi \
    --output_dir outputs/datasets/6f_labeled_1024x640_roi_s2 \
    --scale 2
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SENTINEL = -1.0
NEG_PEAK_THRESHOLD = 0.5  # existing heatmaps below this peak are treated as negatives


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as d:
            return d["heatmap"]
    return np.load(path)


def save_heatmap(path: Path, hm: np.ndarray, suffix: str, dtype) -> None:
    hm = hm.astype(dtype)
    if suffix == ".npz":
        np.savez_compressed(path, heatmap=hm)
    else:
        np.save(path, hm)


def resolve_heatmap_path(directory: Path, stem: str) -> Path | None:
    for ext in (".npz", ".npy"):
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def estimate_sigma_cells(heatmap_paths: list[Path], max_n: int = 300) -> float | None:
    """Estimate Gaussian sigma (in OLD heatmap cells) from peak-1 heatmaps.

    For a 2D Gaussian exp(-r^2 / 2s^2) with peak 1, the grid sum ~= 2*pi*s^2.
    Median over many positives is robust to clipped/edge peaks.
    """
    sigmas: list[float] = []
    for p in heatmap_paths:
        hm = load_heatmap(p).astype(np.float64)
        m = float(hm.max())
        if m < NEG_PEAK_THRESHOLD:
            continue
        hm = hm / m
        s2 = hm.sum() / (2.0 * np.pi)
        if s2 > 1e-6:
            sigmas.append(float(np.sqrt(s2)))
        if len(sigmas) >= max_n:
            break
    return float(np.median(sigmas)) if sigmas else None


def render_gaussian(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--scale", type=int, default=2,
        help="Heatmap resolution multiplier (2 = stride 4 -> 2, 256x160 -> 512x320).",
    )
    ap.add_argument(
        "--sigma_cells", type=float, default=None,
        help="Override the NEW-grid sigma in cells. Default: auto-estimate from the "
        "input heatmaps and scale by --scale to preserve physical peak width.",
    )
    args = ap.parse_args()

    in_dir, out_dir = args.input_dir, args.output_dir
    if not in_dir.is_dir():
        raise SystemExit(f"input_dir not found: {in_dir}")

    src_img = next((in_dir / "images").glob("*"))
    im0 = cv2.imread(str(src_img))
    src_h, src_w = im0.shape[:2]

    hm_paths_all = sorted((in_dir / "heatmaps").glob("*"))
    old_h, old_w = load_heatmap(hm_paths_all[0]).shape
    new_h, new_w = old_h * args.scale, old_w * args.scale
    new_stride_x = src_w / new_w
    new_stride_y = src_h / new_h

    if args.sigma_cells is not None:
        new_sigma = args.sigma_cells
        base = None
    else:
        base = estimate_sigma_cells(hm_paths_all)
        if base is None:
            # Pure-negative dataset (no positive peaks): every heatmap is zeros, so
            # sigma is never used. Fall back to a harmless default and continue.
            print("  [note] no positive heatmaps -> pure-negative set, sigma unused")
            new_sigma = 2.5 * args.scale
        else:
            new_sigma = base * args.scale

    print(
        f"{in_dir.name}: img {src_w}x{src_h} | heatmap {old_w}x{old_h} -> {new_w}x{new_h} "
        f"| sigma(old~{base if base else '?'}) -> new {new_sigma:.3f} cells"
    )

    for sub in ("images", "heatmaps", "labels_yolo"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    src_dtype = load_heatmap(hm_paths_all[0]).dtype
    src_suffix = hm_paths_all[0].suffix

    n_pos = n_neg = n_missing = 0
    with (in_dir / "labels.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        stem = row.get("filename", "").strip()
        if not stem:
            continue
        out_hm = out_dir / "heatmaps" / f"{stem}{src_suffix}"
        is_neg = str(row.get("is_negative", "0")).strip() in ("1", "True", "true")
        cx = float(row.get("center_x") or row.get("x") or SENTINEL)
        cy = float(row.get("center_y") or row.get("y") or SENTINEL)
        if is_neg or cx == SENTINEL or cy == SENTINEL:
            hm = np.zeros((new_h, new_w), dtype=np.float64)
            n_neg += 1
        else:
            hm = render_gaussian(new_h, new_w, cx / new_stride_x, cy / new_stride_y, new_sigma)
            n_pos += 1
        save_heatmap(out_hm, hm, src_suffix, src_dtype)

    # copy images + yolo verbatim (image resolution unchanged)
    for sub in ("images", "labels_yolo"):
        for p in sorted((in_dir / sub).glob("*")):
            shutil.copy2(p, out_dir / sub / p.name)
    shutil.copy2(in_dir / "labels.csv", out_dir / "labels.csv")

    cfg_path = in_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg.update({
        "heatmap_width": new_w, "heatmap_height": new_h,
        "heatmap_size": [new_w, new_h],
        "heatmap_stride": int(round(new_stride_x)),
        "heatmap_sigma": new_sigma,
        "restrided_from": f"{in_dir} (heatmap {old_w}x{old_h} -> {new_w}x{new_h})",
    })
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  wrote {out_dir}: {n_pos} positives, {n_neg} negatives")


if __name__ == "__main__":
    main()
