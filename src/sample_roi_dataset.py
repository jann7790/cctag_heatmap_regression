#!/usr/bin/env python3
"""Sample rotated 1024x640 ROIs from a labeled CCTag dataset (no resize).

Turns a labeled full-frame dataset (e.g. real 4096x2160 captures in
``outputs/datasets/6f_labeled``) into a heatmap-regression training set whose
samples match the synthetic 1024 set (``generated_training_sets_1024``):

  * Positive samples: for each marker, crop a ``roi_width x roi_height`` window
    that places the marker at a RANDOM position, with a small RANDOM rotation
    (rotated-rectangle crop). No scaling -- native pixels are preserved, so
    ``ellipse_a/b`` stay valid and the crop matches the no-resize deployment
    tiling (``src/tile_heatmap.py``).
  * Negative samples: random rotated ROIs that do NOT contain a marker center.

Heatmaps are regenerated per ROI at ``heatmap_stride`` (default 4 -> 256x160 for
a 1024x640 ROI), sigma 3.0, to match the synthetic set. Output is drop-in
compatible with ``src/train_cctag_heatmap_ddp.py`` (images/, heatmaps/ NPZ,
labels_yolo/, labels.csv with 23 columns, config.json).

Example:
    uv run python src/sample_roi_dataset.py \
        --dataset_dir ./outputs/datasets/6f_labeled \
        --output_dir ./outputs/datasets/6f_labeled_1024x640_roi
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# 23-column schema, identical order to src/generate_cctag_dataset.py:37-62.
CSV_HEADER = [
    "filename", "x", "y", "center_x", "center_y",
    "ellipse_cx", "ellipse_cy", "ellipse_a", "ellipse_b", "ellipse_angle_rad",
    "occlusion_ratio",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    "yolo_cx", "yolo_cy", "yolo_w", "yolo_h",
    "is_negative", "negative_mode", "has_visible_marker", "visible_marker_ratio",
    "target_clamped",
]

BORDER_MODES = {
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
}


def gaussian_heatmap(size: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    """2D Gaussian heatmap (height, width) peaked at center=(cx, cy) in cell coords.

    Mirrors generate_gaussian_heatmap in src/generate_cctag_dataset.py:1018."""
    h, w = size
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))


def bbox_to_yolo(bbox: tuple[float, float, float, float], image_size: tuple[int, int]):
    """Axis-aligned bbox -> normalized YOLO (cx, cy, w, h). See
    src/generate_cctag_dataset.py:1141."""
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = bbox
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    cx = x_min + width / 2.0
    cy = y_min + height / 2.0
    return (cx / image_width, cy / image_height, width / image_width, height / image_height)


def rotation_matrix(theta: float) -> np.ndarray:
    """2x2 pure-rotation matrix (scale = 1, no resize)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def affine_place(point: tuple[float, float], target: tuple[float, float], theta: float) -> np.ndarray:
    """2x3 affine M with linear part R(theta) and translation chosen so that
    M @ [point, 1] == target. Crops a rotated window when fed to warpAffine."""
    R = rotation_matrix(theta)
    p = np.array(point, dtype=np.float64)
    t = np.array(target, dtype=np.float64) - R @ p
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], dtype=np.float64)


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine to an (N, 2) array of points -> (N, 2)."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts.astype(np.float64), ones])
    return (M @ homo.T).T


def point_inside(M: np.ndarray, point: tuple[float, float], w: int, h: int) -> bool:
    out = apply_affine(M, np.array([point], dtype=np.float64))[0]
    return 0.0 <= out[0] < w and 0.0 <= out[1] < h


def write_sample(
    stem: str,
    roi: np.ndarray,
    heatmap: np.ndarray,
    row: dict,
    yolo_text: str,
    img_dir: Path,
    hm_dir: Path,
    yolo_dir: Path,
    csv_rows: list,
) -> None:
    cv2.imwrite(str(img_dir / f"{stem}.png"), roi)
    np.savez_compressed(str(hm_dir / f"{stem}.npz"), heatmap=heatmap.astype(np.float16))
    with open(yolo_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write(yolo_text)
    csv_rows.append([row[col] for col in CSV_HEADER])


def negative_row(stem: str) -> dict:
    row = {col: "-1.0000" for col in CSV_HEADER}
    row["filename"] = stem
    row["occlusion_ratio"] = "0.0000"
    row["is_negative"] = "1"
    row["negative_mode"] = "background_roi"
    row["has_visible_marker"] = "0"
    row["visible_marker_ratio"] = "0.000000"
    row["target_clamped"] = "0"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", type=Path, default=Path("outputs/datasets/6f_labeled"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi"))
    ap.add_argument("--roi_width", type=int, default=1024)
    ap.add_argument("--roi_height", type=int, default=640)
    ap.add_argument("--pos_per_marker", type=int, default=4)
    ap.add_argument("--neg_per_frame", type=int, default=3)
    ap.add_argument("--max_rotation_deg", type=float, default=15.0)
    ap.add_argument("--center_margin", type=int, default=64,
                    help="Keep the marker center at least this many px from the ROI edge.")
    ap.add_argument("--heatmap_stride", type=int, default=4)
    ap.add_argument("--heatmap_sigma", type=float, default=3.0)
    ap.add_argument("--border_mode", choices=sorted(BORDER_MODES), default="replicate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    W, H = args.roi_width, args.roi_height
    hm_w, hm_h = W // args.heatmap_stride, H // args.heatmap_stride
    border = BORDER_MODES[args.border_mode]
    max_theta = math.radians(args.max_rotation_deg)
    margin = args.center_margin

    src_csv = args.dataset_dir / "labels.csv"
    if not src_csv.is_file():
        raise FileNotFoundError(src_csv)
    with open(src_csv, newline="", encoding="utf-8") as f:
        src_rows = list(csv.DictReader(f))

    src_cfg = {}
    cfg_path = args.dataset_dir / "config.json"
    if cfg_path.is_file():
        src_cfg = json.loads(cfg_path.read_text())
    frame_w = int(src_cfg.get("image_width", 4096))
    frame_h = int(src_cfg.get("image_height", 2160))

    out_dir = args.output_dir
    img_dir, hm_dir, yolo_dir = out_dir / "images", out_dir / "heatmaps", out_dir / "labels_yolo"
    for d in (img_dir, hm_dir, yolo_dir):
        d.mkdir(parents=True, exist_ok=True)

    csv_rows: list = []
    n_pos = n_neg = n_skip = 0

    for src in src_rows:
        stem = src["filename"]
        img_path = args.dataset_dir / "images" / f"{stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] cannot read {img_path}, skipping")
            continue

        is_negative = int(float(src.get("is_negative", "1")))
        marker_center = None
        if not is_negative:
            cx, cy = float(src["center_x"]), float(src["center_y"])
            if math.isfinite(cx) and math.isfinite(cy):
                marker_center = (cx, cy)

        # ---- positive ROIs ----
        if marker_center is not None:
            ecx, ecy = float(src["ellipse_cx"]), float(src["ellipse_cy"])
            ea, eb = float(src["ellipse_a"]), float(src["ellipse_b"])
            eang = float(src["ellipse_angle_rad"])
            src_bbox_corners = np.array([
                [float(src["bbox_xmin"]), float(src["bbox_ymin"])],
                [float(src["bbox_xmax"]), float(src["bbox_ymin"])],
                [float(src["bbox_xmax"]), float(src["bbox_ymax"])],
                [float(src["bbox_xmin"]), float(src["bbox_ymax"])],
            ], dtype=np.float64)

            for k in range(args.pos_per_marker):
                theta = float(rng.uniform(-max_theta, max_theta))
                u = float(rng.uniform(margin, W - margin))
                v = float(rng.uniform(margin, H - margin))
                M = affine_place(marker_center, (u, v), theta)
                roi = cv2.warpAffine(img, M.astype(np.float32), (W, H), borderMode=border)

                e_out = apply_affine(M, np.array([[ecx, ecy]]))[0]
                corners = apply_affine(M, src_bbox_corners)
                bx0 = float(np.clip(corners[:, 0].min(), 0, W))
                by0 = float(np.clip(corners[:, 1].min(), 0, H))
                bx1 = float(np.clip(corners[:, 0].max(), 0, W))
                by1 = float(np.clip(corners[:, 1].max(), 0, H))
                ycx, ycy, yw, yh = bbox_to_yolo((bx0, by0, bx1, by1), (W, H))

                hm = gaussian_heatmap((hm_h, hm_w), (u / args.heatmap_stride, v / args.heatmap_stride), args.heatmap_sigma)

                out_stem = f"{stem}_pos{k}"
                row = {
                    "filename": out_stem,
                    "x": f"{u:.4f}", "y": f"{v:.4f}",
                    "center_x": f"{u:.4f}", "center_y": f"{v:.4f}",
                    "ellipse_cx": f"{e_out[0]:.4f}", "ellipse_cy": f"{e_out[1]:.4f}",
                    "ellipse_a": f"{ea:.4f}", "ellipse_b": f"{eb:.4f}",
                    "ellipse_angle_rad": f"{eang + theta:.6f}",
                    "occlusion_ratio": f"{float(src.get('occlusion_ratio', 0.0) or 0.0):.4f}",
                    "bbox_xmin": f"{bx0:.4f}", "bbox_ymin": f"{by0:.4f}",
                    "bbox_xmax": f"{bx1:.4f}", "bbox_ymax": f"{by1:.4f}",
                    "yolo_cx": f"{ycx:.6f}", "yolo_cy": f"{ycy:.6f}",
                    "yolo_w": f"{yw:.6f}", "yolo_h": f"{yh:.6f}",
                    "is_negative": "0", "negative_mode": "",
                    "has_visible_marker": str(src.get("has_visible_marker", "1")),
                    "visible_marker_ratio": f"{float(src.get('visible_marker_ratio', 1.0) or 0.0):.6f}",
                    "target_clamped": "0",
                }
                yolo_text = f"0 {ycx:.6f} {ycy:.6f} {yw:.6f} {yh:.6f}\n"
                write_sample(out_stem, roi, hm, row, yolo_text, img_dir, hm_dir, yolo_dir, csv_rows)
                n_pos += 1

        # ---- negative ROIs (background; marker center must be outside) ----
        zero_hm = np.zeros((hm_h, hm_w), dtype=np.float32)
        for k in range(args.neg_per_frame):
            M = None
            for _attempt in range(20):
                theta = float(rng.uniform(-max_theta, max_theta))
                sc = (float(rng.uniform(0, frame_w)), float(rng.uniform(0, frame_h)))
                cand = affine_place(sc, (W / 2.0, H / 2.0), theta)
                if marker_center is None or not point_inside(cand, marker_center, W, H):
                    M = cand
                    break
            if M is None:
                n_skip += 1
                continue
            roi = cv2.warpAffine(img, M.astype(np.float32), (W, H), borderMode=border)
            out_stem = f"{stem}_neg{k}"
            write_sample(out_stem, roi, zero_hm, negative_row(out_stem), "", img_dir, hm_dir, yolo_dir, csv_rows)
            n_neg += 1

    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows)

    config = {
        "generation_type": "cctag_roi_sampled_from_labeled",
        "source_dataset": str(args.dataset_dir),
        "image_width": W,
        "image_height": H,
        "heatmap_stride": args.heatmap_stride,
        "heatmap_width": hm_w,
        "heatmap_height": hm_h,
        "heatmap_sigma": args.heatmap_sigma,
        "pos_per_marker": args.pos_per_marker,
        "neg_per_frame": args.neg_per_frame,
        "max_rotation_deg": args.max_rotation_deg,
        "center_margin": margin,
        "border_mode": args.border_mode,
        "seed": args.seed,
        "num_positive": n_pos,
        "num_negative": n_neg,
        "num_samples": n_pos + n_neg,
        "num_source_frames": len(src_rows),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Done. {n_pos} positive + {n_neg} negative = {n_pos + n_neg} samples")
    if n_skip:
        print(f"  ({n_skip} negative ROIs skipped: could not avoid the marker center)")
    print(f"Output: {out_dir}")
    print(f"  heatmaps: {hm_h}x{hm_w} (stride {args.heatmap_stride}, sigma {args.heatmap_sigma})")


if __name__ == "__main__":
    main()
