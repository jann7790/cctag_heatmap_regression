#!/usr/bin/env python3
"""Render an illustrative figure of detection vs tile-boundary position.

Top panel: best peak (max of the two adjacent tiles) as the vertical cut x=B
sweeps across the marker. Bottom panel: thumbnails of the scene around the
marker with the cut line drawn at representative positions.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import IMAGENET_MEAN, IMAGENET_STD, load_model, resolve_device

CKPT = Path("outputs/runs/experiment_sizehead/best.pt")
IMG = Path("40m_example.png")
OUT = Path("outputs/inference/boundary_figure.png")
CX, CY = 882, 390
WT, HT = 1024, 1080
THR = 0.5


def tile_peak(model, tile, in_w, in_h, device):
    img = cv2.resize(tile, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    t = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    hm = (out[0] if isinstance(out, tuple) else out)
    return float(hm.max().item())


def cut(image, x0, x1, yt, yb):
    H, W = image.shape[:2]
    xs0, xs1 = max(0, x0), min(W, x1)
    ys0, ys1 = max(0, yt), min(H, yb)
    c = image[ys0:ys1, xs0:xs1]
    pl, pr, pt, pb = xs0 - x0, x1 - xs1, ys0 - yt, yb - ys1
    if pl or pr or pt or pb:
        c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return c


def main():
    device = resolve_device(None)
    model, config = load_model(CKPT, device)
    in_w, in_h = config.get("input_width", 640), config.get("input_height", 400)
    image = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yt, yb = CY - HT // 2, CY - HT // 2 + HT

    # ---- fine sweep ----
    Bs = list(range(650, 1101, 10))
    best = []
    for B in Bs:
        pl = tile_peak(model, cut(rgb, B - WT, B, yt, yb), in_w, in_h, device)
        pr = tile_peak(model, cut(rgb, B, B + WT, yt, yb), in_w, in_h, device)
        best.append(max(pl, pr))

    # ---- curve panel ----
    PW, PH = 1280, 440
    ml, mr, mt, mb = 90, 40, 64, 64
    plot = np.full((PH, PW, 3), 255, np.uint8)
    x0a, x1a = ml, PW - mr
    y0a, y1a = PH - mb, mt

    def px(B):
        return int(x0a + (B - Bs[0]) / (Bs[-1] - Bs[0]) * (x1a - x0a))

    def py(v):
        return int(y0a + (v - 0) / (1.0 - 0) * (y1a - y0a))

    # detection band shading (where best>=THR)
    for i in range(len(Bs) - 1):
        if best[i] >= THR or best[i + 1] >= THR:
            cv2.rectangle(plot, (px(Bs[i]), y1a), (px(Bs[i + 1]), y0a), (210, 245, 210), -1)
    # axes
    cv2.line(plot, (x0a, y0a), (x1a, y0a), (0, 0, 0), 2)
    cv2.line(plot, (x0a, y0a), (x0a, y1a), (0, 0, 0), 2)
    # gridlines + y labels
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = py(v)
        cv2.line(plot, (x0a, yy), (x1a, yy), (220, 220, 220), 1)
        cv2.putText(plot, f"{v:.2f}", (x0a - 70, yy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    # threshold line
    cv2.line(plot, (x0a, py(THR)), (x1a, py(THR)), (0, 140, 255), 2, cv2.LINE_AA)
    cv2.putText(plot, "threshold 0.5", (x1a - 220, py(THR) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)
    # marker centre marker
    cv2.line(plot, (px(CX), y1a), (px(CX), y0a), (255, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(plot, "marker centre (B=882)", (px(CX) - 175, y0a - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 60, 60), 2, cv2.LINE_AA)
    # curve
    pts = np.array([[px(B), py(v)] for B, v in zip(Bs, best)], np.int32)
    cv2.polylines(plot, [pts], False, (200, 60, 30), 3, cv2.LINE_AA)
    for B, v in zip(Bs, best):
        cv2.circle(plot, (px(B), py(v)), 3, (200, 60, 30), -1, cv2.LINE_AA)
    # x ticks
    for B in range(700, 1101, 100):
        cv2.line(plot, (px(B), y0a), (px(B), y0a + 6), (0, 0, 0), 2)
        cv2.putText(plot, str(B), (px(B) - 18, y0a + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(plot, "tile boundary x = B (px)", (PW // 2 - 130, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(plot, "best peak", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(plot, "Detection vs tile-boundary position (marker split by the cut)",
                (PW // 2 - 290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    # ---- thumbnail row ----
    sel = [782, 882, 982, 1012, 1042]
    labels = []
    for B in sel:
        pl = tile_peak(model, cut(rgb, B - WT, B, yt, yb), in_w, in_h, device)
        pr = tile_peak(model, cut(rgb, B, B + WT, yt, yb), in_w, in_h, device)
        labels.append(max(pl, pr))
    tx0, tx1, ty0, ty1 = 580, 1180, 230, 560
    thumbs = []
    for B, pk in zip(sel, labels):
        crop = rgb[ty0:ty1, tx0:tx1].copy()
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        det = pk >= THR
        col = (60, 180, 60) if det else (60, 60, 230)
        bx = int((B - tx0) / (tx1 - tx0) * crop.shape[1])
        cv2.line(crop, (bx, 0), (bx, crop.shape[0]), col, 3)
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), col, 3)
        bar = np.full((46, crop.shape[1], 3), 255, np.uint8)
        txt = f"B={B}  peak={pk:.2f}  {'DETECT' if det else 'MISS'}"
        cv2.putText(bar, txt, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        thumbs.append(np.vstack([bar, crop]))
    gap = np.full((thumbs[0].shape[0], 12, 3), 255, np.uint8)
    row = thumbs[0]
    for t in thumbs[1:]:
        row = np.hstack([row, gap, t])
    # scale row to plot width
    scale = PW / row.shape[1]
    row = cv2.resize(row, (PW, int(row.shape[0] * scale)))

    sep = np.full((16, PW, 3), 255, np.uint8)
    fig = np.vstack([plot, sep, row])
    cv2.imwrite(str(OUT), fig)
    print("saved", OUT, fig.shape)


if __name__ == "__main__":
    main()
