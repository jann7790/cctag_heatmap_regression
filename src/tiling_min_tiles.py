#!/usr/bin/env python3
"""Find the minimum number of tiles needed to detect a CCTag in a large image.

Splits the input image into an r x c grid (non-overlapping), resizes each tile
to the model input size, runs inference, and reports the peak per tile. Sweeps
grid configurations in ascending order of total tile count and stops at the
first count where any tile produces a peak >= threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_model,
    resolve_device,
)


def peak_of_tile(model, tile_rgb, in_w, in_h, device) -> float:
    img = cv2.resize(tile_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    heatmap = out[0] if isinstance(out, tuple) else out
    return float(heatmap.max().item())


def eval_grid(model, image_rgb, rows, cols, in_w, in_h, device):
    H, W = image_rgb.shape[:2]
    ys = np.linspace(0, H, rows + 1).round().astype(int)
    xs = np.linspace(0, W, cols + 1).round().astype(int)
    best_peak = 0.0
    best_cell = None
    for r in range(rows):
        for c in range(cols):
            tile = image_rgb[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            p = peak_of_tile(model, tile, in_w, in_h, device)
            if p > best_peak:
                best_peak = p
                best_cell = (r, c, xs[c], ys[r], xs[c + 1], ys[r + 1])
    return best_peak, best_cell


def grids_for_count(n: int, target_ar: float):
    """All (rows, cols) with rows*cols == n, sorted by closeness to target tile AR."""
    out = []
    for rows in range(1, n + 1):
        if n % rows == 0:
            out.append((rows, n // rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_tiles", type=int, default=20)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    in_w = config.get("input_width", 640)
    in_h = config.get("input_height", 400)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W = image.shape[:2]
    target_ar = in_w / in_h
    print(f"image {W}x{H}  model input {in_w}x{in_h}  threshold {args.threshold}\n")

    success = None
    for n in range(1, args.max_tiles + 1):
        for rows, cols in grids_for_count(n, target_ar):
            tile_ar = (W / cols) / (H / rows)
            peak, cell = eval_grid(model, image, rows, cols, in_w, in_h, device)
            scale = max((W / cols) / in_w, (H / rows) / in_h)
            hit = "  <-- DETECT" if peak >= args.threshold else ""
            print(f"n={n:2d}  grid {rows}x{cols:<2d} (tile {W//cols}x{H//rows}, "
                  f"AR {tile_ar:.2f}, downscale {scale:.2f}x)  best_peak={peak:.3f}{hit}")
            if peak >= args.threshold and success is None:
                success = (n, rows, cols, peak, cell)
        if success is not None:
            break

    print()
    if success:
        n, rows, cols, peak, cell = success
        print(f"MIN TILES = {n}  (grid {rows}x{cols}), best_peak={peak:.3f}")
        if cell:
            r, c, x0, y0, x1, y1 = cell
            print(f"detected in cell (row={r}, col={c}) bbox=({x0},{y0})-({x1},{y1})")
    else:
        print(f"No detection up to {args.max_tiles} tiles.")


if __name__ == "__main__":
    main()
