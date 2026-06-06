#!/usr/bin/env python3
"""Sweep ALL grid configurations up to max_tiles, record downscale and peak.
Then render a scatter plot: downscale factor vs best_peak, colour-coded by detection.
Also renders a best-peak-per-N bar chart, and saves a CSV of all results.
"""
from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import IMAGENET_MEAN, IMAGENET_STD, load_model, resolve_device

CKPT = Path("outputs/runs/experiment_sizehead/best.pt")
IMG  = Path("40m_example.png")
THR  = 0.5
MAX  = 48
DW, DH = 320, 200  # thumbnail display size


def peak_of_tile(model, tile_rgb, in_w, in_h, device) -> float:
    img = cv2.resize(tile_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
    t   = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    return float((out[0] if isinstance(out, tuple) else out).max().item())


def eval_grid(model, img_rgb, rows, cols, in_w, in_h, device):
    H, W = img_rgb.shape[:2]
    ys = np.linspace(0, H, rows+1).round().astype(int)
    xs = np.linspace(0, W, cols+1).round().astype(int)
    best = 0.0
    for r in range(rows):
        for c in range(cols):
            tile = img_rgb[ys[r]:ys[r+1], xs[c]:xs[c+1]]
            p = peak_of_tile(model, tile, in_w, in_h, device)
            best = max(best, p)
    return best


def grids_for_n(n):
    return [(r, n//r) for r in range(1, n+1) if n % r == 0]


# ── canvas helpers ──────────────────────────────────────────────────────────

def draw_grid(canvas, xs, ys, x0, x1, y0, y1, color=(220,220,220), lw=1):
    for x in xs:
        cv2.line(canvas, (x, y0), (x, y1), color, lw)
    for y in ys:
        cv2.line(canvas, (x0, y), (x1, y), color, lw)

def px_fn(val, vmin, vmax, x0, x1):
    return int(x0 + (val - vmin) / max(vmax - vmin, 1e-9) * (x1 - x0))

def py_fn(val, vmin, vmax, y0, y1):
    return int(y0 + (val - vmin) / max(vmax - vmin, 1e-9) * (y1 - y0))


def main():
    device = resolve_device(None)
    model, config = load_model(CKPT, device)
    in_w  = config.get("input_width",  640)
    in_h  = config.get("input_height", 400)

    rgb = cv2.cvtColor(cv2.imread(str(IMG)), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    records = []  # (n, rows, cols, tile_w, tile_h, ar, downscale, peak)
    best_per_n = {}

    print(f"{'n':>3} {'grid':>8}  {'tile WxH':>12}  {'AR':>5}  {'ds':>5}  peak")
    for n in range(1, MAX+1):
        for rows, cols in grids_for_n(n):
            tw = W // cols
            th = H // rows
            ar = tw / max(th, 1)
            ds = max(tw / in_w, th / in_h)
            pk = eval_grid(model, rgb, rows, cols, in_w, in_h, device)
            flag = " <DETECT" if pk >= THR else ""
            print(f"{n:>3}  {rows}x{cols:<4}  {tw:>5}x{th:<5}  {ar:>5.2f}  {ds:>5.2f}  {pk:.3f}{flag}")
            records.append((n, rows, cols, tw, th, ar, ds, pk))
            best_per_n[n] = max(best_per_n.get(n, 0.0), pk)

    # ── CSV ─────────────────────────────────────────────────────────────────
    out_csv = Path("outputs/inference/tiling_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n","rows","cols","tile_w","tile_h","aspect_ratio","downscale","peak","detect"])
        for r in records:
            w.writerow(list(r) + [int(r[7] >= THR)])
    print(f"\nCSV saved: {out_csv}")

    # ── Figure 1: downscale vs peak scatter ─────────────────────────────────
    PW, PH = 900, 600
    ml, mr, mt, mb = 70, 30, 50, 60
    fig1 = np.full((PH, PW, 3), 255, np.uint8)
    xa, ya = ml, PH - mb
    xb, yb = PW - mr, mt
    ds_vals = [r[6] for r in records]
    pk_vals = [r[7] for r in records]
    ds_min, ds_max = 0.0, max(ds_vals) + 0.5
    cv2.rectangle(fig1, (xa, yb), (xb, ya), (240,240,240), -1)
    # grid
    for v in np.arange(1, 7, 0.5):
        x = px_fn(v, ds_min, ds_max, xa, xb)
        cv2.line(fig1, (x, yb), (x, ya), (210,210,210), 1)
        cv2.putText(fig1, f"{v:.1f}x", (x-16, ya+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)
    for v in np.arange(0, 1.01, 0.25):
        y = py_fn(v, 0, 1, ya, yb)
        cv2.line(fig1, (xa, y), (xb, y), (210,210,210), 1)
        cv2.putText(fig1, f"{v:.2f}", (xa-52, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)
    # threshold line
    yt = py_fn(THR, 0, 1, ya, yb)
    cv2.line(fig1, (xa, yt), (xb, yt), (0,160,255), 2)
    cv2.putText(fig1, "thr 0.5", (xb-80, yt-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,220), 2)
    # axes
    cv2.line(fig1, (xa, ya), (xb, ya), (0,0,0), 2)
    cv2.line(fig1, (xa, ya), (xa, yb), (0,0,0), 2)
    # scatter points (colour by detect, size by n)
    for n, rows, cols, tw, th, ar, ds, pk in records:
        x = px_fn(ds, ds_min, ds_max, xa, xb)
        y = py_fn(pk, 0, 1, ya, yb)
        col = (60,160,60) if pk >= THR else (220,80,60)
        r = max(4, min(10, int(12 - n * 0.15)))
        cv2.circle(fig1, (x, y), r, col, -1, cv2.LINE_AA)
        cv2.circle(fig1, (x, y), r, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(fig1, "Downscale factor (per-tile)", (PW//2-110, PH-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(fig1, "best peak",   (8, yb+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(fig1, "Downscale vs Peak  (green=DETECT, red=MISS)",
                (ml, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,30,30), 2)
    out1 = Path("outputs/inference/sweep_scatter.png")
    cv2.imwrite(str(out1), fig1)

    # ── Figure 2: best peak per N (bar chart) ───────────────────────────────
    ns = list(range(1, MAX+1))
    bps = [best_per_n.get(n, 0.0) for n in ns]
    PW2, PH2 = 1100, 500
    ml2, mr2, mt2, mb2 = 70, 30, 50, 70
    fig2 = np.full((PH2, PW2, 3), 255, np.uint8)
    xa2, ya2 = ml2, PH2 - mb2
    xb2, yb2 = PW2 - mr2, mt2
    bar_w = (xb2 - xa2) // (MAX + 2)
    for i, (n, bp) in enumerate(zip(ns, bps)):
        bx = xa2 + i * bar_w + bar_w // 4
        by = py_fn(bp, 0, 1, ya2, yb2)
        col = (60,160,60) if bp >= THR else (220,80,60)
        cv2.rectangle(fig2, (bx, by), (bx + bar_w//2, ya2), col, -1)
        cv2.rectangle(fig2, (bx, by), (bx + bar_w//2, ya2), (0,0,0), 1)
        # tick label
        if n % 4 == 0 or n == 1 or n == 8:
            cv2.putText(fig2, str(n), (bx, ya2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
    # threshold
    yt2 = py_fn(THR, 0, 1, ya2, yb2)
    cv2.line(fig2, (xa2, yt2), (xb2, yt2), (0,160,255), 2)
    cv2.putText(fig2, "thr 0.5", (xb2-80, yt2-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,220), 2)
    # y grid
    for v in np.arange(0, 1.01, 0.25):
        y = py_fn(v, 0, 1, ya2, yb2)
        cv2.line(fig2, (xa2, y), (xb2, y), (210,210,210), 1)
        cv2.putText(fig2, f"{v:.2f}", (xa2-52, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)
    cv2.line(fig2, (xa2, ya2), (xb2, ya2), (0,0,0), 2)
    cv2.line(fig2, (xa2, ya2), (xa2, yb2), (0,0,0), 2)
    cv2.putText(fig2, "Number of tiles (N)",  (PW2//2-100, PH2-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(fig2, "best peak", (8, yb2+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(fig2, "Best peak across all grids per tile count N  (green=DETECT, red=MISS)",
                (ml2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 2)
    out2 = Path("outputs/inference/sweep_bar.png")
    cv2.imwrite(str(out2), fig2)

    # ── combined ────────────────────────────────────────────────────────────
    f1 = cv2.imread(str(out1)); f2 = cv2.imread(str(out2))
    f2r = cv2.resize(f2, (f1.shape[1], int(f2.shape[0] * f1.shape[1] / f2.shape[1])))
    sep = np.full((16, f1.shape[1], 3), 255, np.uint8)
    combo = np.vstack([f1, sep, f2r])
    out3 = Path("outputs/inference/sweep_combined.png")
    cv2.imwrite(str(out3), combo)
    print(f"scatter:  {out1}\nbar:      {out2}\ncombined: {out3}")


if __name__ == "__main__":
    main()
