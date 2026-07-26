#!/usr/bin/env python3
"""
Generate a boundary-crop visibility sweep test set.

Goal: empirically answer "how much of a CCTag can be cut off by a tile/frame
edge and still be detected?" -- the question that sets the tile overlap for a
4K tiled-inference deployment.

Method (CPU only, no GPU):
  - take fully-in-frame real markers from a clean ROI dataset,
  - cut out the marker disc with a feathered elliptical mask (real texture),
  - resize it to a controlled apparent diameter (deployment apparent size),
  - composite it onto a real background crop so a controlled fraction `v` of the
    disc is inside the frame, with the rest clipped by ONE frame edge.

The cut is by the IMAGE BORDER -- this is exactly what a tile boundary does to a
straddling tag (the missing part lives in the neighbour tile). visible_ratio is
defined on a circle model (markers here are near-circular, a/b ~ 0.95).

Output: images/ + labels.csv + config.json + montage.png (QC).
Each row records the controlled (target_diam, visible_ratio, cut_dir), the true
marker centre in canvas coords (may be off-frame), and the visible-part bbox so
the runner can tell a real hit from a background false positive.

Example:
  uv run python src/make_visibility_sweep.py \
    --roi_dir ./outputs/datasets/6f_labeled_1024x640_roi \
    --output_dir ./outputs/datasets/visibility_sweep \
    --diams 100 140 200 --visibilities 0.25 0.4 0.5 0.6 0.7 0.85 1.0 \
    --samples_per_cell 40 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 1024, 640
CUT_DIRS = ("L", "R", "T", "B")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCTag boundary-crop visibility sweep generator")
    p.add_argument("--roi_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi"),
                   help="Clean ROI dataset with images/ and labels.csv (marker source + backgrounds)")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/datasets/visibility_sweep"))
    p.add_argument("--diams", type=float, nargs="+", default=[100.0, 140.0, 200.0],
                   help="Target marker major-axis diameters (px, apparent in the 1024x640 input)")
    p.add_argument("--visibilities", type=float, nargs="+",
                   default=[0.25, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0],
                   help="Visible fractions of the disc to sweep (1.0 = fully in-frame)")
    p.add_argument("--samples_per_cell", type=int, default=40,
                   help="Samples per (diam, visibility) cell; cut direction randomised within a cell")
    p.add_argument("--directions", type=str, nargs="+", default=list(CUT_DIRS),
                   choices=list(CUT_DIRS), help="Which frame edges may do the cutting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame_w", type=int, default=FRAME_W)
    p.add_argument("--frame_h", type=int, default=FRAME_H)
    return p.parse_args()


def fraction_visible(t: float, r: float) -> float:
    """Fraction of a radius-r disc with local-x <= t (t in [-r, r])."""
    if t <= -r:
        return 0.0
    if t >= r:
        return 1.0
    seg = r * r * math.acos(t / r) - t * math.sqrt(max(r * r - t * t, 0.0))  # area with x > t
    return 1.0 - seg / (math.pi * r * r)


def invert_fraction(v: float, r: float) -> float:
    """Cut offset t (from centre) giving visible fraction v; monotonic in t."""
    lo, hi = -r, r
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fraction_visible(mid, r) < v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_rows(roi_dir: Path) -> tuple[list[dict], list[dict]]:
    rows = list(csv.DictReader(open(roi_dir / "labels.csv")))

    def is_neg(r: dict) -> bool:
        return r.get("is_negative", "").lower() in ("1", "true")

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def full_in(r: dict, margin_scale: float = 1.15) -> bool:
        cx, cy = fl(r["center_x"]), fl(r["center_y"])
        a, b = fl(r["ellipse_a"]), fl(r["ellipse_b"])
        if None in (cx, cy, a, b):
            return False
        rad = max(a, b) * margin_scale
        return cx - rad > 0 and cy - rad > 0 and cx + rad < FRAME_W and cy + rad < FRAME_H

    pos = [r for r in rows if not is_neg(r) and full_in(r)]
    neg = [r for r in rows if is_neg(r)]
    return pos, neg


def cutout_marker(img: np.ndarray, row: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (bgr_patch, alpha[0..1], src_radius) cropped tight around the marker."""
    cx, cy = float(row["center_x"]), float(row["center_y"])
    a, b = float(row["ellipse_a"]), float(row["ellipse_b"])
    angle = math.degrees(float(row.get("ellipse_angle_rad", 0.0) or 0.0))
    rad = max(a, b)
    half = int(math.ceil(rad * 1.15))
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    patch = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    # elliptical alpha mask, feathered
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    ctr = (half, half)
    cv2.ellipse(mask, ctr, (int(round(a * 1.03)), int(round(b * 1.03))),
                angle, 0, 360, 255, -1, cv2.LINE_AA)
    feather = max(3, int(round(rad * 0.03)) | 1)  # odd kernel
    alpha = cv2.GaussianBlur(mask, (feather, feather), 0).astype(np.float32) / 255.0
    return patch, alpha, rad


def composite(canvas: np.ndarray, patch: np.ndarray, alpha: np.ndarray,
              cx: float, cy: float) -> None:
    """Alpha-blend patch onto canvas centred at (cx, cy), clipping at borders."""
    ph, pw = patch.shape[:2]
    px0 = int(round(cx - pw / 2.0))
    py0 = int(round(cy - ph / 2.0))
    H, W = canvas.shape[:2]
    # intersection of patch rect with canvas
    dx0, dy0 = max(0, px0), max(0, py0)
    dx1, dy1 = min(W, px0 + pw), min(H, py0 + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sx0, sy0 = dx0 - px0, dy0 - py0
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    a = alpha[sy0:sy1, sx0:sx1, None]
    canvas[dy0:dy1, dx0:dx1] = (
        a * patch[sy0:sy1, sx0:sx1] + (1.0 - a) * canvas[dy0:dy1, dx0:dx1]
    ).astype(np.uint8)


def main() -> None:
    args = parse_args()
    global FRAME_W, FRAME_H
    FRAME_W, FRAME_H = args.frame_w, args.frame_h
    rng = random.Random(args.seed)

    pos, neg = load_rows(args.roi_dir)
    if not pos or not neg:
        raise SystemExit(f"need positives and negatives in {args.roi_dir} (got {len(pos)} pos, {len(neg)} neg)")
    print(f"source: {len(pos)} full-in-frame markers, {len(neg)} backgrounds")

    img_dir = args.output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_cache: dict[str, np.ndarray] = {}

    def load_img(stem: str) -> np.ndarray | None:
        if stem not in img_cache:
            img = cv2.imread(str(args.roi_dir / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
            img_cache[stem] = img
        return img_cache[stem]

    label_rows: list[dict] = []
    montage_cells: dict[tuple[float, float], np.ndarray] = {}
    idx = 0
    for diam in args.diams:
        r_t = diam / 2.0
        for v in args.visibilities:
            made = 0
            attempts = 0
            while made < args.samples_per_cell and attempts < args.samples_per_cell * 20:
                attempts += 1
                src = rng.choice(pos)
                bg = rng.choice(neg)
                simg = load_img(src["filename"])
                bgimg = load_img(bg["filename"])
                if simg is None or bgimg is None:
                    continue
                patch, alpha, rad = cutout_marker(simg, src)
                if patch.size == 0 or min(patch.shape[:2]) < 4:
                    continue
                scale = diam / (2.0 * rad)
                new_w = max(4, int(round(patch.shape[1] * scale)))
                new_h = max(4, int(round(patch.shape[0] * scale)))
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                patch_r = cv2.resize(patch, (new_w, new_h), interpolation=interp)
                alpha_r = cv2.resize(alpha, (new_w, new_h), interpolation=interp)

                canvas = cv2.resize(bgimg, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                cut = rng.choice(args.directions)
                t = invert_fraction(v, r_t)  # signed offset of cut line from centre
                # place centre so the chosen frame edge cuts the disc to fraction v
                if cut == "R":
                    cx = FRAME_W - t
                    cy = rng.uniform(r_t, FRAME_H - r_t)
                elif cut == "L":
                    cx = t
                    cy = rng.uniform(r_t, FRAME_H - r_t)
                elif cut == "B":
                    cy = FRAME_H - t
                    cx = rng.uniform(r_t, FRAME_W - r_t)
                else:  # "T"
                    cy = t
                    cx = rng.uniform(r_t, FRAME_W - r_t)

                composite(canvas, patch_r, alpha_r, cx, cy)
                actual_v = fraction_visible(t, r_t)

                # visible-part bbox (disc clipped to frame)
                vx0 = max(0.0, cx - r_t); vy0 = max(0.0, cy - r_t)
                vx1 = min(float(FRAME_W), cx + r_t); vy1 = min(float(FRAME_H), cy + r_t)
                in_frame = 1 if (0 <= cx < FRAME_W and 0 <= cy < FRAME_H) else 0

                fname = f"sweep_{idx:06d}_d{int(diam)}_v{int(round(actual_v*100)):03d}_{cut}.png"
                cv2.imwrite(str(img_dir / fname), canvas)
                label_rows.append({
                    "filename": fname,
                    "src_filename": src["filename"],
                    "bg_filename": bg["filename"],
                    "target_diam": f"{diam:.1f}",
                    "visible_ratio": f"{actual_v:.4f}",
                    "cut_dir": cut,
                    "true_cx": f"{cx:.2f}",
                    "true_cy": f"{cy:.2f}",
                    "center_in_frame": in_frame,
                    "vis_xmin": f"{vx0:.1f}", "vis_ymin": f"{vy0:.1f}",
                    "vis_xmax": f"{vx1:.1f}", "vis_ymax": f"{vy1:.1f}",
                })
                montage_cells.setdefault((diam, v), canvas.copy())
                idx += 1
                made += 1
            print(f"  diam={diam:.0f} v={v:.2f}: {made} images")

    # write labels.csv
    with open(args.output_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        w.writeheader()
        w.writerows(label_rows)

    # config.json
    with open(args.output_dir / "config.json", "w") as f:
        json.dump({
            "roi_dir": str(args.roi_dir),
            "frame_w": FRAME_W, "frame_h": FRAME_H,
            "diams": args.diams, "visibilities": args.visibilities,
            "samples_per_cell": args.samples_per_cell,
            "directions": args.directions, "seed": args.seed,
            "total_images": len(label_rows),
            "note": "boundary-crop detection sweep; cut by image border = tile-edge cut",
        }, f, indent=2)

    # montage: rows = diams, cols = visibilities (one example per cell)
    cell_w, cell_h = 256, 160
    rows_img = []
    for diam in args.diams:
        cols = []
        for v in args.visibilities:
            c = montage_cells.get((diam, v))
            tile = cv2.resize(c, (cell_w, cell_h)) if c is not None else np.zeros((cell_h, cell_w, 3), np.uint8)
            cv2.putText(tile, f"d{int(diam)} v{v:.2f}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cols.append(tile)
        rows_img.append(np.hstack(cols))
    cv2.imwrite(str(args.output_dir / "montage.png"), np.vstack(rows_img))

    print(f"\nwrote {len(label_rows)} images to {img_dir}")
    print(f"labels: {args.output_dir/'labels.csv'}")
    print(f"QC montage: {args.output_dir/'montage.png'}")


if __name__ == "__main__":
    main()
