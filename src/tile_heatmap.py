"""Tiled heatmap (CNN) inference overlay on large images.

Tiles a big image into native windows the size of the model's training input
(default 640x400, 1:1 no resize), runs the heatmap model per tile, decodes the
peak, maps it back to full-image coordinates, and draws it.

Usage:
  uv run python src/tile_heatmap.py \
    --checkpoint outputs/runs/stride4_offw1.0/best.pt \
    --source img.png --output outputs/inference/hm_tiled --threshold 0.4
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# load the infer module to reuse load_model + constants + decode
_spec = importlib.util.spec_from_file_location("hm", str(Path(__file__).with_name("infer_cctag_heatmap.py")))
hm = importlib.util.module_from_spec(_spec)
sys.modules["hm"] = hm
_spec.loader.exec_module(hm)


def starts(total: int, tile: int, overlap: float):
    if tile >= total:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    pos = list(range(0, total - tile + 1, step))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="outputs/inference/hm_tiled")
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--min_peak_sharpness", type=float, default=3.0,
                    help="Reject peaks whose peak/neighbourhood ratio is below this "
                         "(real CCTags are sharp >3; bright-edge FPs are diffuse). 0 disables.")
    ap.add_argument("--no_offset", action="store_true",
                    help="Ignore the offset head (use argmax+parabolic sub-pixel only).")
    ap.add_argument("--max_size_frac", type=float, default=1.2,
                    help="Reject a peak if size-head diameter exceeds this fraction of the tile "
                         "(implausible = bright-edge FP). 0 disables.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, config = hm.load_model(args.checkpoint, device)
    tw = int(config.get("input_width", 640))
    th = int(config.get("input_height", 400))
    mean = hm.IMAGENET_MEAN.to(device)
    std = hm.IMAGENET_STD.to(device)

    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(args.source)
    H, W = img.shape[:2]
    full = img.copy()

    xs, ys = starts(W, tw, args.overlap), starts(H, th, args.overlap)
    dets = []  # (gx, gy, peak, radius_px)
    n_sharp_rej = n_size_rej = 0
    with torch.inference_mode():
        for y0 in ys:
            for x0 in xs:
                tile = img[y0:y0 + th, x0:x0 + tw]
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().to(device) / 255.0
                t = ((t - mean) / std).unsqueeze(0)
                out = model(t)
                # model heads: (heatmap[, offset][, size])
                outs = out if isinstance(out, tuple) else (out,)
                heatmap = outs[0][0, 0].float().cpu().numpy()
                offset = outs[1][0].float().cpu().numpy() if len(outs) >= 2 else None
                size_log = outs[2][0].float().cpu().numpy() if len(outs) >= 3 else None

                peak = float(heatmap.max())
                if peak < args.threshold:
                    continue
                # (1) sharpness filter: drop diffuse bright-edge false positives
                if args.min_peak_sharpness > 0:
                    if hm.compute_peak_sharpness(heatmap) < args.min_peak_sharpness:
                        n_sharp_rej += 1
                        continue
                # (2) sub-pixel center via offset head (fallback: parabolic subpixel)
                if offset is not None and not args.no_offset:
                    res = hm.decode_center_offset(heatmap, offset, threshold=args.threshold)
                else:
                    res = hm.decode_center_subpixel(heatmap, threshold=args.threshold)
                if res is None:
                    continue
                hx, hy = res
                sx, sy = tw / heatmap.shape[1], th / heatmap.shape[0]
                # (3) marker radius from size head (source-image px); also implausibility filter
                radius_px = 14
                if size_log is not None:
                    sz = hm.decode_size_at_peak(heatmap, size_log, threshold=args.threshold)
                    if sz is not None:
                        a, b = sz  # ellipse semi-axes in tile px
                        diam = 2.0 * max(a, b)
                        if args.max_size_frac > 0 and diam > args.max_size_frac * tw:
                            n_size_rej += 1
                            continue
                        radius_px = int(max(a, b))
                gx, gy = int(hx * sx + x0), int(hy * sy + y0)
                dets.append((gx, gy, peak, radius_px))

    # dedup: merge peaks within 25px, keep highest
    dets.sort(key=lambda d: -d[2])
    kept = []
    for gx, gy, p, r in dets:
        if all((gx - kx) ** 2 + (gy - ky) ** 2 > 25 ** 2 for kx, ky, _, _ in kept):
            kept.append((gx, gy, p, r))

    for gx, gy, p, r in kept:
        cv2.circle(full, (gx, gy), max(6, r), (0, 0, 255), 2)
        cv2.drawMarker(full, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(full, f"{p:.2f}", (gx + r + 4, gy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = out_dir / (Path(args.source).stem + "_hm.jpg")
    cv2.imwrite(str(out_path), full)
    print(f"peaks: {len(kept)}  tiles: {len(xs)}x{len(ys)}  "
          f"(sharp_rej={n_sharp_rej} size_rej={n_size_rej})  Saved: {out_path}")


if __name__ == "__main__":
    main()
