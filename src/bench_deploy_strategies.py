#!/usr/bin/env python3
"""Benchmark full-frame deployment strategies for continuous newest-frame inference.

Compares, on real 4096x2160 frames (no YOLO stage):
  full4k  - native 4K input, decoder output overridden to stride-4 (1024x540)
  tile16  - 16x 1024x640 native tiles (4 exact cols, 4 rows / 133px y-overlap), batched
  tile30  - 30%-overlap 1024x640 tiling (tile_heatmap.py default, 30 tiles), batched
  roi     - single 1024x640 crop (tracking-mode reference, timing only)

Reports per-strategy latency (preprocess / forward / decode, CUDA events) and
accuracy on the 6f_labeled full frames: detection rate on positives, false-positive
rate on negatives, and center L2 in 4K pixels (detected positives only).

Usage:
  CUDA_VISIBLE_DEVICES=4 uv run python src/bench_deploy_strategies.py \
    --checkpoint outputs/runs/lower_l2_err/best.pt \
    --dataset_dir outputs/datasets/6f_labeled --limit 300
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from infer_cctag_heatmap import (
    IMAGENET_MEAN, IMAGENET_STD,
    decode_center_offset, load_model, resolve_device,
)

STRIDE = 4


def starts(total: int, tile: int, overlap: float) -> list[int]:
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(total - tile, 0) + 1, step))
    if xs[-1] != total - tile:
        xs.append(total - tile)
    return xs


def tile_origins(W: int, H: int, tw: int, th: int, overlap: float) -> list[tuple[int, int]]:
    return [(x, y) for y in starts(H, th, overlap) for x in starts(W, tw, overlap)]


def normalize_frame(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Full frame BGR uint8 -> normalized float tensor (1,3,H,W) on device."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


def set_output_size(model: torch.nn.Module, hm_h: int, hm_w: int) -> None:
    model.heatmap_height = hm_h
    model.heatmap_width = hm_w


def run_model(model, batch: torch.Tensor):
    with torch.no_grad():
        out = model(batch)
    if isinstance(out, tuple):
        hm, offset = out[0], out[1]
    else:
        hm, offset = out, None
    return hm, offset


def decode_full4k(hm_t, off_t, threshold: float):
    """Returns (peak, x_frame, y_frame) for the dynamic full-frame output."""
    hm = hm_t[0, 0].float().cpu().numpy()
    peak = float(hm.max())
    off = off_t[0].float().cpu().numpy() if off_t is not None else None
    if off is not None:
        res = decode_center_offset(hm, off, threshold)
    else:
        res = None
    if res is None:
        return peak, None, None
    return peak, res[0] * STRIDE, res[1] * STRIDE


def decode_tiles(hm_t, off_t, origins, threshold: float):
    """Pick the tile with the global max peak; returns (peak, x_frame, y_frame)."""
    hms = hm_t[:, 0].float().cpu().numpy()
    peaks = hms.reshape(hms.shape[0], -1).max(axis=1)
    best = int(peaks.argmax())
    peak = float(peaks[best])
    off = off_t[best].float().cpu().numpy() if off_t is not None else None
    res = decode_center_offset(hms[best], off, threshold) if off is not None else None
    if res is None:
        return peak, None, None
    ox, oy = origins[best]
    return peak, res[0] * STRIDE + ox, res[1] * STRIDE + oy


def bench_latency(model, frame_bgr, strategies, device, n_warmup=5, n_bench=20,
                  threshold=0.5, amp=False):
    H, W = frame_bgr.shape[:2]
    rows = []
    for name, cfg in strategies.items():
        pre_ms, fwd_ms, dec_ms = [], [], []
        for i in range(n_warmup + n_bench):
            t0 = time.perf_counter()
            full = normalize_frame(frame_bgr, device)
            if cfg["mode"] == "full":
                set_output_size(model, H // STRIDE, W // STRIDE)
                batch = full
                origins = None
            else:
                set_output_size(model, cfg["th"] // STRIDE, cfg["tw"] // STRIDE)
                origins = cfg["origins"]
                batch = torch.cat(
                    [full[:, :, y:y + cfg["th"], x:x + cfg["tw"]] for x, y in origins]
                )
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            if amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    hm, off = run_model(model, batch)
            else:
                hm, off = run_model(model, batch)
            ev1.record()
            torch.cuda.synchronize(device)

            t2 = time.perf_counter()
            if cfg["mode"] == "full":
                decode_full4k(hm, off, threshold)
            else:
                decode_tiles(hm, off, origins, threshold)
            t3 = time.perf_counter()

            if i >= n_warmup:
                pre_ms.append((t1 - t0) * 1e3)
                fwd_ms.append(ev0.elapsed_time(ev1))
                dec_ms.append((t3 - t2) * 1e3)
        rows.append({
            "strategy": name,
            "tiles": 1 if cfg["mode"] == "full" else len(cfg["origins"]),
            "pre_ms": float(np.mean(pre_ms)),
            "fwd_ms": float(np.mean(fwd_ms)),
            "dec_ms": float(np.mean(dec_ms)),
            "total_ms": float(np.mean(pre_ms) + np.mean(fwd_ms) + np.mean(dec_ms)),
        })
    return rows


def eval_accuracy(model, rows_gt, images_dir: Path, strategies, device,
                  threshold=0.5, limit=0):
    results = {name: [] for name in strategies}
    rows_gt = rows_gt[:limit] if limit else rows_gt
    for i, gt in enumerate(rows_gt):
        img_path = images_dir / gt["filename"]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        full = normalize_frame(bgr, device)
        is_neg = gt["is_negative"] == "1" or gt["is_negative"].lower() == "true"
        gx = None if is_neg else float(gt["center_x"])
        gy = None if is_neg else float(gt["center_y"])

        for name, cfg in strategies.items():
            if cfg["mode"] == "full":
                set_output_size(model, H // STRIDE, W // STRIDE)
                hm, off = run_model(model, full)
                peak, px, py = decode_full4k(hm, off, threshold)
            else:
                set_output_size(model, cfg["th"] // STRIDE, cfg["tw"] // STRIDE)
                origins = cfg["origins"]
                batch = torch.cat(
                    [full[:, :, y:y + cfg["th"], x:x + cfg["tw"]] for x, y in origins]
                )
                hm, off = run_model(model, batch)
                peak, px, py = decode_tiles(hm, off, origins, threshold)
            detected = peak >= threshold and px is not None
            l2 = None
            if detected and not is_neg:
                l2 = float(np.hypot(px - gx, py - gy))
            results[name].append({"is_neg": is_neg, "detected": detected, "l2": l2})
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(rows_gt)}] evaluated", flush=True)
    return results


def summarize_accuracy(results):
    rows = []
    for name, recs in results.items():
        pos = [r for r in recs if not r["is_neg"]]
        neg = [r for r in recs if r["is_neg"]]
        det = [r for r in pos if r["detected"]]
        l2s = np.array([r["l2"] for r in det], dtype=np.float64)
        rows.append({
            "strategy": name,
            "n_pos": len(pos),
            "n_neg": len(neg),
            "det_rate": len(det) / len(pos) if pos else float("nan"),
            "fp_rate": (sum(r["detected"] for r in neg) / len(neg)) if neg else float("nan"),
            "l2_mean": float(l2s.mean()) if len(l2s) else float("nan"),
            "l2_median": float(np.median(l2s)) if len(l2s) else float("nan"),
            "l2_p95": float(np.percentile(l2s, 95)) if len(l2s) else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("outputs/runs/lower_l2_err/best.pt"))
    ap.add_argument("--dataset_dir", type=Path,
                    default=Path("outputs/datasets/6f_labeled"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=300,
                    help="Max frames for the accuracy eval (0 = all)")
    ap.add_argument("--amp", action="store_true",
                    help="Also report fp16 autocast latency")
    ap.add_argument("--skip_accuracy", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, config = load_model(args.checkpoint, device)
    tw = int(config.get("input_width", 1024))
    th = int(config.get("input_height", 640))
    print(f"Model: {config.get('backbone')}  trained input {tw}x{th}, "
          f"heatmap {config.get('heatmap_width')}x{config.get('heatmap_height')}")

    labels = args.dataset_dir / "labels.csv"
    with open(labels, newline="") as f:
        rows_gt = list(csv.DictReader(f))
    images_dir = args.dataset_dir / "images"
    for gt in rows_gt:
        if not Path(gt["filename"]).suffix:
            gt["filename"] += ".png"
    sample = cv2.imread(str(images_dir / rows_gt[0]["filename"]), cv2.IMREAD_COLOR)
    H, W = sample.shape[:2]
    print(f"Frame size: {W}x{H}, {len(rows_gt)} labeled frames")

    strategies = {
        "full4k": {"mode": "full"},
        "tile16": {"mode": "tile", "tw": tw, "th": th,
                   "origins": tile_origins(W, H, tw, th, overlap=0.0)},
        "tile30": {"mode": "tile", "tw": tw, "th": th,
                   "origins": tile_origins(W, H, tw, th, overlap=0.3)},
    }
    timing_strategies = dict(strategies)
    timing_strategies["roi"] = {"mode": "tile", "tw": tw, "th": th,
                                "origins": [((W - tw) // 2, (H - th) // 2)]}
    for name, cfg in timing_strategies.items():
        n = 1 if cfg["mode"] == "full" else len(cfg["origins"])
        print(f"  {name}: {n} forward window(s)")

    print("\n=== Latency (fp32) ===")
    rows = bench_latency(model, sample, timing_strategies, device,
                         threshold=args.threshold)
    hdr = f"{'strategy':10s} {'tiles':>5s} {'pre':>8s} {'fwd':>8s} {'dec':>8s} {'total':>8s} {'fps':>6s}"
    print(hdr)
    for r in rows:
        print(f"{r['strategy']:10s} {r['tiles']:5d} {r['pre_ms']:7.1f}m {r['fwd_ms']:7.1f}m "
              f"{r['dec_ms']:7.1f}m {r['total_ms']:7.1f}m {1000.0 / r['total_ms']:6.1f}")

    if args.amp:
        print("\n=== Latency (fp16 autocast) ===")
        rows = bench_latency(model, sample, timing_strategies, device,
                             threshold=args.threshold, amp=True)
        print(hdr)
        for r in rows:
            print(f"{r['strategy']:10s} {r['tiles']:5d} {r['pre_ms']:7.1f}m {r['fwd_ms']:7.1f}m "
                  f"{r['dec_ms']:7.1f}m {r['total_ms']:7.1f}m {1000.0 / r['total_ms']:6.1f}")

    if not args.skip_accuracy:
        n = args.limit if args.limit else len(rows_gt)
        print(f"\n=== Accuracy on {min(n, len(rows_gt))} frames (fp32, thr={args.threshold}) ===")
        results = eval_accuracy(model, rows_gt, images_dir, strategies, device,
                                threshold=args.threshold, limit=args.limit)
        print(f"{'strategy':10s} {'pos':>5s} {'neg':>5s} {'det%':>7s} {'fp%':>7s} "
              f"{'L2mean':>8s} {'L2med':>8s} {'L2p95':>8s}")
        for r in summarize_accuracy(results):
            print(f"{r['strategy']:10s} {r['n_pos']:5d} {r['n_neg']:5d} "
                  f"{100 * r['det_rate']:6.1f}% {100 * r['fp_rate']:6.1f}% "
                  f"{r['l2_mean']:7.2f}px {r['l2_median']:7.2f}px {r['l2_p95']:7.2f}px")


if __name__ == "__main__":
    main()
