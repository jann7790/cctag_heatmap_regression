#!/usr/bin/env python3
"""Test inference at different input resolutions by bypassing the fixed 640x400 resize.
The model is fully convolutional (ResNet18 U-Net, no FC), so it accepts any W×H.
The heatmap output is always interpolated to (heatmap_h=100, heatmap_w=160).

Tested resolutions (W×H):
  640×400   – trained size (baseline)
  1280×800  – 2× trained size
  1920×1200 – 3×
  2048×1280 – 3.2×
  4096×2160 – native (no resize at all)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import (
    IMAGENET_MEAN, IMAGENET_STD,
    decode_center_offset, decode_center_weighted, decode_size_at_peak,
    load_model, resolve_device,
)

CKPT    = Path("outputs/runs/experiment_sizehead/best.pt")
IMG     = Path("40m_example.png")
OUTDIR  = Path("outputs/inference/input_res_test")
THR     = 0.5
RESOLUTIONS = [
    (640,  400,  "640×400  (trained, baseline)"),
    (1280, 800,  "1280×800  (2×)"),
    (1920, 1200, "1920×1200 (3×)"),
    (2048, 1280, "2048×1280 (3.2×)"),
    (4096, 2160, "4096×2160 (native, no resize)"),
]
DISPLAY_W, DISPLAY_H = 640, 400  # all overlays shown at this size


def run_at_res(model, bgr_src, in_w, in_h, device):
    """Resize image to in_w×in_h, run model, return (overlay_bgr, peak, center_src_xy)."""
    rgb = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)

    t = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    t = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)

    if isinstance(out, tuple):
        hm     = out[0][0, 0].cpu().numpy()
        offset = out[1][0].cpu().numpy() if out[1] is not None else None
        size   = out[2][0].cpu().numpy() if len(out) > 2 and out[2] is not None else None
    else:
        hm, offset, size = out[0, 0].cpu().numpy(), None, None

    peak = float(hm.max())

    # ── overlay (drawn on display-size version of the resized input) ──
    disp_bgr = cv2.resize(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
                          (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
    hm_u8    = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(
        cv2.resize(hm_u8, (DISPLAY_W, DISPLAY_H)), cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(disp_bgr, 0.55, hm_color, 0.45, 0)

    center_src = None
    if peak >= THR:
        hm_h, hm_w = hm.shape
        if offset is not None:
            c_hm = decode_center_offset(hm, offset, threshold=THR)
        else:
            c_hm = decode_center_weighted(hm, threshold=THR)
        if c_hm is not None:
            # convert heatmap coords → display coords
            dx = int(c_hm[0] * DISPLAY_W / hm_w)
            dy = int(c_hm[1] * DISPLAY_H / hm_h)
            # convert heatmap coords → source image coords
            src_h, src_w = bgr_src.shape[:2]
            sx = c_hm[0] * src_w / hm_w
            sy = c_hm[1] * src_h / hm_h
            center_src = (sx, sy)

            if size is not None:
                ab = decode_size_at_peak(hm, size, threshold=THR)
                if ab and ab[0] > 0 and ab[1] > 0:
                    # scale ellipse from model-input px → display px
                    scale_x = DISPLAY_W / in_w
                    scale_y = DISPLAY_H / in_h
                    axes = (max(1, int(ab[0] * scale_x)), max(1, int(ab[1] * scale_y)))
                    cv2.ellipse(overlay, (dx, dy), axes, 0, 0, 360, (0, 255, 255), 2)
            cv2.circle(overlay, (dx, dy), 5, (0, 255, 0), -1)
            cv2.circle(overlay, (dx, dy), 7, (0, 0, 0), 2)

    return overlay, peak, center_src, hm


def make_label_bar(text, peak, detected, w=DISPLAY_W):
    col   = (50, 180, 50) if detected else (50, 50, 210)
    bar   = np.full((52, w, 3), 255, np.uint8)
    cv2.putText(bar, text,
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30, 30, 30), 2, cv2.LINE_AA)
    status = f"peak={peak:.3f}  {'DETECT' if detected else 'MISS'}"
    cv2.putText(bar, status,
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2, cv2.LINE_AA)
    return bar


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(None)
    model, config = load_model(CKPT, device)

    src = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    src_h, src_w = src.shape[:2]
    print(f"source image: {src_w}×{src_h}\n")
    print(f"{'resolution':<24} {'downscale':>10}  {'peak':>6}  {'src center':>20}  result")

    panels = []
    for in_w, in_h, label in RESOLUTIONS:
        ds = max(src_w / in_w, src_h / in_h)
        overlay, peak, center_src, hm = run_at_res(model, src, in_w, in_h, device)
        detected = peak >= THR
        cstr = f"({center_src[0]:.0f}, {center_src[1]:.0f})" if center_src else "—"
        print(f"{label:<24}  {ds:>8.2f}×  {peak:>6.3f}  {cstr:>20}  {'DETECT' if detected else 'MISS'}")

        cv2.imwrite(str(OUTDIR / f"overlay_{in_w}x{in_h}.png"), overlay)
        # save raw heatmap (false-colour only, no blend) for comparison
        hm_u8    = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(cv2.resize(hm_u8, (DISPLAY_W, DISPLAY_H)), cv2.COLORMAP_JET)
        cv2.imwrite(str(OUTDIR / f"heatmap_{in_w}x{in_h}.png"), hm_color)

        bar = make_label_bar(label, peak, detected)
        cv2.rectangle(overlay, (0, 0), (DISPLAY_W-1, DISPLAY_H-1),
                      (50,180,50) if detected else (50,50,210), 3)
        panels.append(np.vstack([bar, overlay]))

    # ── 2-row grid (3 top, 2 bottom centred) ──
    gap_v = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    row1  = np.hstack([panels[0], gap_v, panels[1], gap_v, panels[2]])
    pad_w = (row1.shape[1] - panels[3].shape[1] * 2 - 12) // 2
    pad   = np.full((panels[3].shape[0], pad_w, 3), 255, np.uint8)
    row2  = np.hstack([pad, panels[3], gap_v, panels[4], pad])
    sep   = np.full((16, row1.shape[1], 3), 255, np.uint8)

    # title bar
    title = np.full((48, row1.shape[1], 3), 255, np.uint8)
    cv2.putText(title,
                "Inference at different input resolutions  (same 4K source, same model weights)",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 30, 30), 2, cv2.LINE_AA)

    fig = np.vstack([title, row1, sep, row2])
    out = Path("outputs/inference/input_res_comparison.png")
    cv2.imwrite(str(out), fig)
    print(f"\nfigure saved: {out}  shape={fig.shape}")


if __name__ == "__main__":
    main()
