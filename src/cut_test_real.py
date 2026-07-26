#!/usr/bin/env python3
"""
Boundary-cut tolerance test on REAL images, using the model's own detection.

For each real image:
  1. run the model on the full frame -> detected centre C (ground-truth anchor),
  2. estimate the tag outer radius R (radial oscillation scan; override with --radius),
  3. crop a window around C whose ONE edge cuts the tag to a controlled visible
     fraction v (circle model), from each of 4 edges -- this is exactly what a tile
     boundary does to a straddling tag,
  4. re-run the model on each cut crop (resized to the model input) and record
     whether it still fires (peak) and how far the new centre drifts from C.

Outputs per image a montage (rows = cut directions, cols = visibility) annotated
with peak / centre-error, plus a CSV and an aggregated table.

GPU. Example:
  CUDA_VISIBLE_DEVICES=3 uv run python src/cut_test_real.py \
    --checkpoint outputs/runs/fable_occ_hrnet_w18/best.pt \
    --input testset --output outputs/inference/testset_cut --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_cctag_heatmap import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, load_model,
    decode_center_offset, decode_center_softargmax,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boundary-cut tolerance test on real images")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="image file or directory")
    p.add_argument("--output", type=Path, default=Path("outputs/inference/testset_cut"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.3, help="peak detection threshold")
    p.add_argument("--visibilities", type=float, nargs="+",
                   default=[0.25, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0])
    p.add_argument("--directions", type=str, nargs="+", default=["L", "R", "T", "B"],
                   choices=["L", "R", "T", "B"])
    p.add_argument("--radius", type=float, default=None,
                   help="override estimated tag outer radius (px, original coords) for ALL images")
    p.add_argument("--window_scale", type=float, default=3.0,
                   help="crop window side = window_scale * R (context around the tag)")
    p.add_argument("--softargmax_radius", type=int, default=5)
    return p.parse_args()


def fraction_visible(t: float, r: float) -> float:
    if t <= -r:
        return 0.0
    if t >= r:
        return 1.0
    seg = r * r * math.acos(t / r) - t * math.sqrt(max(r * r - t * t, 0.0))
    return 1.0 - seg / (math.pi * r * r)


def invert_fraction(v: float, r: float) -> float:
    lo, hi = -r, r
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fraction_visible(mid, r) < v:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_radius(gray: np.ndarray, cx: float, cy: float, rmax: int) -> float:
    """Outer ring radius via the radial mean-intensity oscillation envelope.

    CCTag concentric rings make the angle-averaged intensity oscillate with r and
    flatten past the outer ring. We take the largest r where the local oscillation
    is still a meaningful fraction of its peak. Robust-ish to partial occlusion
    because it averages over all angles.
    """
    H, W = gray.shape
    angles = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    radii = np.arange(6, rmax, 2)
    prof = np.zeros(len(radii), np.float32)
    for i, r in enumerate(radii):
        xs = np.clip((cx + r * cos).astype(int), 0, W - 1)
        ys = np.clip((cy + r * sin).astype(int), 0, H - 1)
        prof[i] = gray[ys, xs].mean()
    # local oscillation = rolling std of the derivative magnitude
    d = np.abs(np.diff(prof, prepend=prof[0]))
    win = 5
    osc = np.array([d[max(0, i - win):i + win + 1].std() for i in range(len(d))])
    if osc.max() < 1e-6:
        return rmax * 0.4
    thr = 0.20 * osc.max()
    idx = np.where(osc > thr)[0]
    return float(radii[idx[-1]]) if len(idx) else rmax * 0.4


def to_input(bgr: np.ndarray, in_w: int, in_h: int, device: torch.device) -> torch.Tensor:
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


def run(model, config, bgr, device, thr, sa_radius):
    """Return (peak, center_xy_in_bgr_coords or None, heatmap)."""
    in_w, in_h = config["input_width"], config["input_height"]
    hm_w, hm_h = config["heatmap_width"], config["heatmap_height"]
    x = to_input(bgr, in_w, in_h, device)
    with torch.no_grad():
        out = model(x)
    offset = None
    if isinstance(out, tuple):
        heatmap_t = out[0]
        if len(out) >= 2 and out[1] is not None:
            offset = out[1][0].float().cpu().numpy()
    else:
        heatmap_t = out
    heatmap = heatmap_t[0, 0].float().cpu().numpy()
    peak = float(heatmap.max())
    if offset is not None:
        res = decode_center_offset(heatmap, offset, threshold=thr, soft_peak=True,
                                   radius=sa_radius, temperature=1.0)
    else:
        res = decode_center_softargmax(heatmap, threshold=thr, radius=sa_radius)
    center = None
    if res is not None:
        hx, hy = res
        bh, bw = bgr.shape[:2]
        center = (hx / hm_w * bw, hy / hm_h * bh)
    return peak, center, heatmap


def crop_cut(img, cx, cy, R, v, direction, in_w, in_h):
    """Trim the FULL frame at a cut line through the tag, then pad to model aspect.

    Keeps the full framing the model already detects on (no zoom), removes only the
    cut-off side of the tag (exactly a tile boundary), and pads the short axis with
    black to the model aspect so the tag is NOT anisotropically stretched.
    Returns (model_input_bgr, transform) where transform maps model coords back to
    original-image coords.
    """
    H, W = img.shape[:2]
    t = invert_fraction(v, R)  # signed offset of cut line from centre
    xa, ya, xb, yb = 0, 0, W, H
    if direction == "R":
        xb = int(round(cx + t))
    elif direction == "L":
        xa = int(round(cx - t))
    elif direction == "B":
        yb = int(round(cy + t))
    else:  # T
        ya = int(round(cy - t))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(W, xb), min(H, yb)
    if xb - xa < 8 or yb - ya < 8:
        return None, None
    crop = img[ya:yb, xa:xb]
    ch, cw = crop.shape[:2]
    ar = in_w / in_h
    if cw / ch < ar:
        pad_w = int(round(ch * ar)) - cw
        pl, pr, pt, pb = pad_w // 2, pad_w - pad_w // 2, 0, 0
    else:
        pad_h = int(round(cw / ar)) - ch
        pl, pr, pt, pb = 0, 0, pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (xa, ya, pl, pt, padded.shape[1], padded.shape[0])


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    config.setdefault("input_width", 1024); config.setdefault("input_height", 640)
    config.setdefault("heatmap_width", 256); config.setdefault("heatmap_height", 160)
    print(f"model input {config['input_width']}x{config['input_height']} on {device}")

    if args.input.is_dir():
        paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        paths = [args.input]
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip unreadable {path.name}")
            continue
        H, W = img.shape[:2]
        peak0, c0, _ = run(model, config, img, device, args.threshold, args.softargmax_radius)
        if c0 is None:
            print(f"{path.name}: no detection on full frame, skipping")
            continue
        cx, cy = c0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rmax = int(min(cx, cy, W - cx, H - cy, max(W, H) * 0.45))
        R = args.radius if args.radius else estimate_radius(gray, cx, cy, max(rmax, 40))
        print(f"\n{path.name}: full peak={peak0:.3f} center=({cx:.0f},{cy:.0f}) R~{R:.0f}px")

        cells = {}
        in_w, in_h = config["input_width"], config["input_height"]
        for d in args.directions:
            for v in args.visibilities:
                crop, tf = crop_cut(img, cx, cy, R, v, d, in_w, in_h)
                if crop is None:
                    continue
                peak, cc, _ = run(model, config, crop, device, args.threshold, args.softargmax_radius)
                err = None
                if cc is not None and tf is not None:
                    xa, ya, pl, pt, pw, ph = tf
                    # model coords -> padded -> trimmed -> original
                    ox = xa + (cc[0] / crop.shape[1] * pw - pl)
                    oy = ya + (cc[1] / crop.shape[0] * ph - pt)
                    err = math.hypot(ox - cx, oy - cy)
                detected = peak >= args.threshold
                rows.append({"image": path.name, "direction": d,
                             "visible_ratio": round(v, 3), "peak": round(peak, 4),
                             "detected": int(detected),
                             "center_err_px": "" if err is None else round(err, 1)})
                # build annotated cell
                cell = cv2.resize(crop, (256, 160))
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(cell, f"{d} v{v:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
                lab2 = f"p{peak:.2f}" + (f" e{err:.0f}" if err is not None else "")
                cv2.putText(cell, lab2, (4, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)
                cells[(d, v)] = cell

        # montage rows=directions cols=visibilities
        mrows = []
        for d in args.directions:
            cols = [cells.get((d, v), np.zeros((160, 256, 3), np.uint8)) for v in args.visibilities]
            mrows.append(np.hstack(cols))
        cv2.imwrite(str(args.output / f"{path.stem}_cutmontage.png"), np.vstack(mrows))

    with open(args.output / "cut_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "direction", "visible_ratio",
                                          "peak", "detected", "center_err_px"])
        w.writeheader(); w.writerows(rows)

    # aggregate per visibility (over images+directions)
    print("\n=== detection rate & median center-error vs visibility (all images/dirs) ===")
    print(f"{'visible':>8} {'det_rate':>9} {'med_peak':>9} {'med_err_px':>11}  n")
    vs = sorted({r["visible_ratio"] for r in rows})
    for v in vs:
        sub = [r for r in rows if r["visible_ratio"] == v]
        det = sum(r["detected"] for r in sub) / len(sub)
        peaks = sorted(r["peak"] for r in sub)
        errs = sorted(float(r["center_err_px"]) for r in sub if r["center_err_px"] != "")
        med_peak = peaks[len(peaks) // 2]
        med_err = errs[len(errs) // 2] if errs else float("nan")
        print(f"{v:>8.2f} {det:>9.0%} {med_peak:>9.3f} {med_err:>11.1f}  {len(sub)}")
    print(f"\nmontages + cut_results.csv in {args.output}")


if __name__ == "__main__":
    main()
