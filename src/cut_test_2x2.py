#!/usr/bin/env python3
"""
2x2 tiling corner-cut tolerance test (synthetic, scale-controlled).

The 2x2 (4-tile) worst case: a tag near the tile cross-junction is split by TWO
perpendicular tile edges, so each tile sees only a CORNER quadrant (~25% at the
junction). This is harder than a single-edge (2-tile) cut, which shows a half-disc
segment. This script measures it directly.

Geometry: a 2048x1280 canvas = 2x2 of the model input (1024x640). Tiles are sliced
1:1 and fed to the model with NO resize, so marker diameter == apparent diameter
(no scale confound). A real marker is composited near the cross-junction (canvas
centre) at a controlled offset; we sweep that offset from 0 (tag exactly on the
junction -> 25% per quadrant) outward (tag falls into one tile -> 100%).

Metrics per frame:
  any-of-4   : at least one of the 4 tiles fires (peak >= threshold) -- the real
               pipeline result (you run all 4 tiles).
  best-tile  : the tile holding the most of the tag (max visible fraction) fires.

GPU. Example:
  CUDA_VISIBLE_DEVICES=3 uv run python src/cut_test_2x2.py \
    --checkpoint outputs/runs/fable_occ_hrnet_w18/best.pt \
    --output outputs/inference/cut2x2 --device cuda:0 \
    --diams 100 200 320 --samples_per_cell 30
"""
from __future__ import annotations

import argparse
import csv
import math
import random
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
from make_visibility_sweep import cutout_marker, load_rows, composite  # noqa: E402

TILE_W, TILE_H = 1024, 640
CANVAS_W, CANVAS_H = 2 * TILE_W, 2 * TILE_H  # 2048x1280
JUNCTION = (TILE_W, TILE_H)                  # cross at canvas centre
TILES = {                                    # (x0,y0,x1,y1)
    "TL": (0, 0, TILE_W, TILE_H),
    "TR": (TILE_W, 0, CANVAS_W, TILE_H),
    "BL": (0, TILE_H, TILE_W, CANVAS_H),
    "BR": (TILE_W, TILE_H, CANVAS_W, CANVAS_H),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2x2 corner-cut tolerance test")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--roi_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi"))
    p.add_argument("--output", type=Path, default=Path("outputs/inference/cut2x2"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--diams", type=float, nargs="+", default=[100.0, 200.0, 320.0],
                   help="apparent marker diameters (px), fed 1:1 into a 1024x640 tile")
    p.add_argument("--offsets", type=float, nargs="+",
                   default=[0.0, 0.35, 0.7, 1.05, 1.4, 1.8],
                   help="junction->centre offset as fraction of radius (0 = tag on junction)")
    p.add_argument("--samples_per_cell", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--softargmax_radius", type=int, default=5)
    return p.parse_args()


def to_input(bgr, device):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.shape[1] != TILE_W or img.shape[0] != TILE_H:
        img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


def run(model, config, bgr, device, thr, sa_radius):
    hm_w, hm_h = config["heatmap_width"], config["heatmap_height"]
    x = to_input(bgr, device)
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
        center = (res[0] / hm_w * bgr.shape[1], res[1] / hm_h * bgr.shape[0])
    return peak, center


def tile_fractions(cx, cy, r):
    """Exact disc-area fraction inside each of the 4 tiles."""
    mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    total = float(mask.sum()) / 255.0
    fr = {}
    for name, (x0, y0, x1, y1) in TILES.items():
        fr[name] = (float(mask[y0:y1, x0:x1].sum()) / 255.0) / total if total > 0 else 0.0
    return fr


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    config.setdefault("heatmap_width", 256); config.setdefault("heatmap_height", 160)
    rng = random.Random(args.seed)
    pos, neg = load_rows(args.roi_dir)
    print(f"model on {device}; {len(pos)} source markers, {len(neg)} backgrounds")
    args.output.mkdir(parents=True, exist_ok=True)
    img_cache = {}

    def load_img(stem):
        if stem not in img_cache:
            img_cache[stem] = cv2.imread(str(args.roi_dir / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
        return img_cache[stem]

    rows = []
    montage = {}  # (diam, offset) -> list of 4 annotated tiles
    for diam in args.diams:
        r_t = diam / 2.0
        for off in args.offsets:
            m = off * r_t
            ox = oy = m / math.sqrt(2.0)          # diagonal, toward BR
            cx, cy = JUNCTION[0] + ox, JUNCTION[1] + oy
            made = attempts = 0
            while made < args.samples_per_cell and attempts < args.samples_per_cell * 20:
                attempts += 1
                src = rng.choice(pos); bg = rng.choice(neg)
                simg, bgimg = load_img(src["filename"]), load_img(bg["filename"])
                if simg is None or bgimg is None:
                    continue
                patch, alpha, rad = cutout_marker(simg, src)
                if patch.size == 0 or min(patch.shape[:2]) < 4:
                    continue
                s = diam / (2.0 * rad)
                nw, nh = max(4, int(patch.shape[1] * s)), max(4, int(patch.shape[0] * s))
                interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
                patch_r = cv2.resize(patch, (nw, nh), interpolation=interp)
                alpha_r = cv2.resize(alpha, (nw, nh), interpolation=interp)
                canvas = cv2.resize(bgimg, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
                composite(canvas, patch_r, alpha_r, cx, cy)
                fr = tile_fractions(cx, cy, r_t)

                peaks, dets, tinfo = {}, {}, {}
                for name, (x0, y0, x1, y1) in TILES.items():
                    tile = canvas[y0:y1, x0:x1]
                    peak, cc = run(model, config, tile, device, args.threshold, args.softargmax_radius)
                    peaks[name] = peak
                    # honest detection: peak fires AND lands ON the composited marker
                    # (decoded centre within one radius of the true marker centre, in
                    #  canvas coords) -- rejects background false positives.
                    gx = gy = None
                    on_marker = False
                    if peak >= args.threshold and cc is not None:
                        gx, gy = x0 + cc[0], y0 + cc[1]
                        if fr[name] > 0.01 and math.hypot(gx - cx, gy - cy) <= r_t:
                            on_marker = True
                    dets[name] = int(on_marker)
                    tinfo[name] = (peak, gx, gy, on_marker)
                best_name = max(fr, key=fr.get)            # tile holding most of the tag
                any_det = int(any(dets.values()))
                best_det = dets[best_name]
                rows.append({
                    "diam": diam, "offset_frac": off,
                    "best_quadrant_frac": round(fr[best_name], 3),
                    "any_of_4_det": any_det, "best_tile_det": best_det,
                    "best_tile": best_name,
                    **{f"peak_{n}": round(peaks[n], 3) for n in TILES},
                    **{f"frac_{n}": round(fr[n], 3) for n in TILES},
                })
                if (diam, off) not in montage:  # keep first sample's full canvas for the explainer
                    montage[(diam, off)] = (canvas.copy(), cx, cy, r_t, tinfo, fr[best_name], any_det)
                made += 1
            print(f"  diam={diam:.0f} offset={off:.2f}r: {made} frames")

    with open(args.output / "cut2x2_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # explainer per diam: one full frame per offset row, with the 2x2 cross, the
    # true tag (orange circle) and where each tile fired (green=hit, red=fired-off).
    sc = 760.0 / CANVAS_W
    vw, vh = int(CANVAS_W * sc), int(CANVAS_H * sc)
    for diam in args.diams:
        legend = np.zeros((30, vw, 3), np.uint8)
        cv2.putText(legend, f"diam {int(diam)}px  |  cyan=tile cross   orange=true tag   "
                    f"green dot=tile detected ON tag   red dot=fired off-tag",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panels = [legend]
        for off in args.offsets:
            item = montage.get((diam, off))
            if item is None:
                continue
            canvas, cx, cy, rt, tinfo, bq, any_det = item
            vis = cv2.resize(canvas, (vw, vh))
            cv2.line(vis, (int(TILE_W * sc), 0), (int(TILE_W * sc), vh), (255, 255, 0), 1)
            cv2.line(vis, (0, int(TILE_H * sc)), (vw, int(TILE_H * sc)), (255, 255, 0), 1)
            cv2.circle(vis, (int(cx * sc), int(cy * sc)), max(3, int(rt * sc)), (0, 200, 255), 2)
            for _, (peak, gx, gy, on) in tinfo.items():
                if gx is not None:
                    cv2.circle(vis, (int(gx * sc), int(gy * sc)), 7,
                               (0, 255, 0) if on else (0, 0, 255), 2)
            banner = np.zeros((26, vw, 3), np.uint8)
            cv2.putText(banner, f"offset {off:.2f}r   best-quadrant {bq*100:.0f}% visible   "
                        f"any-of-4: {'YES' if any_det else 'NO'}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0) if any_det else (0, 0, 255), 1, cv2.LINE_AA)
            panels.append(banner)
            panels.append(vis)
        cv2.imwrite(str(args.output / f"cut2x2_explain_d{int(diam)}.png"), np.vstack(panels))

    # summary tables
    print("\n=== 2x2 corner-cut: detection vs offset (per diam) ===")
    for diam in args.diams:
        print(f"\n--- apparent diameter {int(diam)}px ---")
        print(f"{'offset(r)':>10} {'best_quad%':>11} {'any-of-4':>9} {'best-tile':>10}  n")
        for off in args.offsets:
            sub = [r for r in rows if r["diam"] == diam and r["offset_frac"] == off]
            if not sub:
                continue
            bq = sum(r["best_quadrant_frac"] for r in sub) / len(sub)
            anyd = sum(r["any_of_4_det"] for r in sub) / len(sub)
            bestd = sum(r["best_tile_det"] for r in sub) / len(sub)
            print(f"{off:>10.2f} {bq:>10.0%} {anyd:>9.0%} {bestd:>10.0%}  {len(sub)}")
    print(f"\nresults: {args.output/'cut2x2_results.csv'}; montages: {args.output}/cut2x2_d*.png")


if __name__ == "__main__":
    main()
