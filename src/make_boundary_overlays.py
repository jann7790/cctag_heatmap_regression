#!/usr/bin/env python3
"""For each tile-boundary position B, render the inference overlay of BOTH
adjacent tiles (left = source [B-Wt, B], right = [B, B+Wt]).

Each overlay is the model's 640x400 input with the predicted heatmap blended
(JET) and the decoded centre/ellipse drawn when peak >= threshold. Tiles are
arranged as one row per B, two columns (left | right).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    decode_center_offset,
    decode_center_weighted,
    decode_size_at_peak,
    load_model,
    resolve_device,
)

CKPT = Path("outputs/runs/experiment_sizehead/best.pt")
IMG = Path("40m_example.png")
OUTDIR = Path("outputs/inference/boundary_overlays")
CX, CY = 882, 390
WT, HT = 1024, 1080
THR = 0.5
B_LIST = [682, 782, 882, 962, 982, 1002, 1022, 1042]
DW, DH = 384, 240  # display size per overlay


def cut(image, x0, x1, yt, yb):
    H, W = image.shape[:2]
    xs0, xs1 = max(0, x0), min(W, x1)
    ys0, ys1 = max(0, yt), min(H, yb)
    c = image[ys0:ys1, xs0:xs1]
    pl, pr, pt, pb = xs0 - x0, x1 - xs1, ys0 - yt, yb - ys1
    if pl or pr or pt or pb:
        c = cv2.copyMakeBorder(c, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return c


def infer_overlay(model, tile_rgb, in_w, in_h, device):
    """Return (overlay_bgr at in_w x in_h, peak)."""
    disp = cv2.resize(tile_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)  # RGB
    t = torch.from_numpy(disp.transpose(2, 0, 1)).float() / 255.0
    t = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    if isinstance(out, tuple):
        hm = out[0][0, 0].cpu().numpy()
        offset = out[1][0].cpu().numpy() if out[1] is not None else None
        size = out[2][0].cpu().numpy() if len(out) > 2 and out[2] is not None else None
    else:
        hm = out[0, 0].cpu().numpy()
        offset, size = None, None
    peak = float(hm.max())

    base = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    hm_u8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(cv2.resize(hm_u8, (in_w, in_h)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.6, hm_color, 0.4, 0)

    if peak >= THR:
        hm_h, hm_w = hm.shape
        if offset is not None:
            c = decode_center_offset(hm, offset, threshold=THR)
        else:
            c = decode_center_weighted(hm, threshold=THR)
        if c is not None:
            cx = int(c[0] * in_w / hm_w)
            cy = int(c[1] * in_h / hm_h)
            if size is not None:
                ab = decode_size_at_peak(hm, size, threshold=THR)
                if ab and ab[0] > 0 and ab[1] > 0:
                    cv2.ellipse(overlay, (cx, cy), (int(ab[0]), int(ab[1])), 0, 0, 360, (0, 255, 255), 2)
            cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (cx, cy), 7, (0, 0, 0), 2)
    return overlay, peak


def labeled(overlay, text, detect):
    o = cv2.resize(overlay, (DW, DH))
    col = (60, 180, 60) if detect else (60, 60, 230)
    cv2.rectangle(o, (0, 0), (DW - 1, DH - 1), col, 3)
    bar = np.full((34, DW, 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return np.vstack([bar, o])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(None)
    model, config = load_model(CKPT, device)
    in_w, in_h = config.get("input_width", 640), config.get("input_height", 400)
    rgb = cv2.cvtColor(cv2.imread(str(IMG), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    yt, yb = CY - HT // 2, CY - HT // 2 + HT

    rows = []
    # header row: column titles
    hdr = np.full((40, DW * 2 + 12, 3), 255, np.uint8)
    cv2.putText(hdr, "LEFT tile  [B-1024, B]", (60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(hdr, "RIGHT tile  [B, B+1024]", (DW + 70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    rows.append(hdr)

    gap = np.full((DH + 34, 12, 3), 255, np.uint8)
    for B in B_LIST:
        ol, pl = infer_overlay(model, cut(rgb, B - WT, B, yt, yb), in_w, in_h, device)
        orr, pr = infer_overlay(model, cut(rgb, B, B + WT, yt, yb), in_w, in_h, device)
        cv2.imwrite(str(OUTDIR / f"B{B}_left.png"), ol)
        cv2.imwrite(str(OUTDIR / f"B{B}_right.png"), orr)
        lL = labeled(ol, f"B={B} L peak={pl:.3f} {'DETECT' if pl >= THR else 'MISS'}", pl >= THR)
        lR = labeled(orr, f"B={B} R peak={pr:.3f} {'DETECT' if pr >= THR else 'MISS'}", pr >= THR)
        rows.append(np.hstack([lL, gap, lR]))
        print(f"B={B:4d}  left={pl:.3f}  right={pr:.3f}")

    sep = lambda: np.full((10, rows[1].shape[1], 3), 255, np.uint8)
    stacked = [rows[0]]
    for r in rows[1:]:
        stacked.extend([sep(), r])
    fig = np.vstack(stacked)
    out = Path("outputs/inference/boundary_overlays_grid.png")
    cv2.imwrite(str(out), fig)
    print("saved", out, fig.shape)


if __name__ == "__main__":
    main()
