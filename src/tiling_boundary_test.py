#!/usr/bin/env python3
"""Test detection when a CCTag is split across a tile boundary.

Two adjacent tiles of fixed size (Wt x Ht) share a vertical boundary at x=B.
Left tile  = source [B-Wt, B], right tile = source [B, B+Wt]. As B sweeps across
the marker centre, the marker is progressively cut between the two tiles. We
report the peak each tile produces and the best of the two.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import IMAGENET_MEAN, IMAGENET_STD, load_model, resolve_device


def peak_and_center(model, tile_rgb, in_w, in_h, device):
    img = cv2.resize(tile_rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    hm = (out[0] if isinstance(out, tuple) else out)[0, 0].cpu().numpy()
    return float(hm.max())


def cut_tile(image, x_left, x_right, y_top, y_bot, Wt, Ht):
    """Extract source [x_left:x_right, y_top:y_bot]; pad to Wt x Ht on the
    OUTER side (clamped edge) using edge replication so the boundary cut is exact."""
    H, W = image.shape[:2]
    xs0, xs1 = max(0, x_left), min(W, x_right)
    ys0, ys1 = max(0, y_top), min(H, y_bot)
    crop = image[ys0:ys1, xs0:xs1]
    pad_l = xs0 - x_left      # >0 if x_left < 0 (pad on left)
    pad_r = x_right - xs1      # >0 if x_right > W (pad on right)
    pad_t = ys0 - y_top
    pad_b = y_bot - ys1
    if pad_l or pad_r or pad_t or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--cx", type=float, required=True, help="marker centre x in full image")
    ap.add_argument("--cy", type=float, required=True, help="marker centre y in full image")
    ap.add_argument("--tile_w", type=int, default=1024)
    ap.add_argument("--tile_h", type=int, default=1080)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--span", type=int, default=160, help="sweep B over cx +/- span")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    in_w = config.get("input_width", 640)
    in_h = config.get("input_height", 400)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    Wt, Ht = args.tile_w, args.tile_h
    y_top = int(round(args.cy - Ht / 2))
    y_bot = y_top + Ht
    cx = args.cx

    print(f"marker centre=({args.cx:.0f},{args.cy:.0f})  tile={Wt}x{Ht}  "
          f"vertical boundary B sweeps {cx-args.span:.0f}..{cx+args.span:.0f}\n")
    print(f"{'B (cut x)':>10} {'offset':>7} {'left_peak':>10} {'right_peak':>11} {'best':>7}  result")
    for B in range(int(cx - args.span), int(cx + args.span) + 1, args.step):
        left = cut_tile(image, B - Wt, B, y_top, y_bot, Wt, Ht)
        right = cut_tile(image, B, B + Wt, y_top, y_bot, Wt, Ht)
        pl = peak_and_center(model, left, in_w, in_h, device)
        pr = peak_and_center(model, right, in_w, in_h, device)
        best = max(pl, pr)
        res = "DETECT" if best >= args.threshold else "miss"
        marker = "  <-- cut through centre" if B == int(cx) or abs(B - cx) < args.step / 2 else ""
        print(f"{B:>10d} {B-cx:>+7.0f} {pl:>10.3f} {pr:>11.3f} {best:>7.3f}  {res}{marker}")


if __name__ == "__main__":
    main()
