#!/usr/bin/env python3
"""Measure CPU and GPU inference latency at each input resolution.

Reports:
  - preprocess time   (imread + resize + normalize)
  - model forward time (GPU: CUDA events; CPU: perf_counter)
  - total time
for N_WARMUP warmup runs followed by N_BENCH timed runs.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_cctag_heatmap import IMAGENET_MEAN, IMAGENET_STD, load_model, resolve_device

CKPT       = Path("outputs/runs/experiment_sizehead/best.pt")
IMG        = Path("40m_example.png")
THR        = 0.5
N_WARMUP   = 5
N_BENCH    = 20
RESOLS     = [
    (640,  400,  "640×400  (trained)"),
    (1280, 800,  "1280×800 (2×)"),
    (1920, 1200, "1920×1200 (3×)"),
    (2048, 1280, "2048×1280 (3.2×)"),
    (4096, 2160, "4096×2160 (native)"),
]


def make_tensor(bgr_src, in_w, in_h, device):
    rgb     = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    t = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
    return t.to(device)


def bench_gpu(model, bgr_src, in_w, in_h, device):
    """Returns (pre_ms, fwd_ms, total_ms) – mean over N_BENCH runs."""
    pre_times, fwd_times = [], []
    for i in range(N_WARMUP + N_BENCH):
        # ── preprocess ──
        t0 = time.perf_counter()
        tensor = make_tensor(bgr_src, in_w, in_h, device)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        pre_ms = (t1 - t0) * 1000

        # ── forward (timed with CUDA events) ──
        e_start = torch.cuda.Event(enable_timing=True)
        e_end   = torch.cuda.Event(enable_timing=True)
        e_start.record()
        with torch.no_grad():
            _ = model(tensor)
        e_end.record()
        torch.cuda.synchronize(device)
        fwd_ms = e_start.elapsed_time(e_end)

        if i >= N_WARMUP:
            pre_times.append(pre_ms)
            fwd_times.append(fwd_ms)

    mp  = float(np.mean(pre_times))
    mf  = float(np.mean(fwd_times))
    return mp, mf, mp + mf


def bench_cpu(model, bgr_src, in_w, in_h, device):
    pre_times, fwd_times = [], []
    for i in range(N_WARMUP + N_BENCH):
        t0 = time.perf_counter()
        tensor = make_tensor(bgr_src, in_w, in_h, device)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        with torch.no_grad():
            _ = model(tensor)
        t3 = time.perf_counter()

        if i >= N_WARMUP:
            pre_times.append((t1 - t0) * 1000)
            fwd_times.append((t3 - t2) * 1000)

    mp = float(np.mean(pre_times))
    mf = float(np.mean(fwd_times))
    return mp, mf, mp + mf


def run_peak(model, bgr_src, in_w, in_h, device):
    t = make_tensor(bgr_src, in_w, in_h, device)
    with torch.no_grad():
        out = model(t)
    hm = (out[0] if isinstance(out, tuple) else out)
    return float(hm.max().item())


def section(title):
    print(f"\n{'═'*72}")
    print(f"  {title}")
    print(f"{'═'*72}")
    print(f"{'Resolution':<26} {'pre(ms)':>9} {'fwd(ms)':>9} {'total(ms)':>10}  peak  result")
    print(f"{'─'*72}")


def main():
    src    = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    gpu_device = resolve_device(None)      # auto → CUDA if available
    cpu_device = torch.device("cpu")

    print(f"benchmark: {N_WARMUP} warmup + {N_BENCH} timed runs per resolution")
    print(f"GPU: {torch.cuda.get_device_name(gpu_device) if gpu_device.type=='cuda' else 'N/A'}")

    # ── GPU ─────────────────────────────────────────────────────────────────
    if gpu_device.type == "cuda":
        section("GPU inference")
        model_gpu, _ = load_model(CKPT, gpu_device)
        model_gpu.eval()
        for in_w, in_h, label in RESOLS:
            pk = run_peak(model_gpu, src, in_w, in_h, gpu_device)
            mp, mf, mt = bench_gpu(model_gpu, src, in_w, in_h, gpu_device)
            det = "DETECT" if pk >= THR else "MISS"
            print(f"{label:<26}  {mp:>8.1f}  {mf:>8.1f}  {mt:>9.1f}  {pk:.3f}  {det}")

    # ── CPU ─────────────────────────────────────────────────────────────────
    section("CPU inference")
    model_cpu, _ = load_model(CKPT, cpu_device)
    model_cpu.eval()
    for in_w, in_h, label in RESOLS:
        pk = run_peak(model_cpu, src, in_w, in_h, cpu_device)
        mp, mf, mt = bench_cpu(model_cpu, src, in_w, in_h, cpu_device)
        det = "DETECT" if pk >= THR else "MISS"
        print(f"{label:<26}  {mp:>8.1f}  {mf:>8.1f}  {mt:>9.1f}  {pk:.3f}  {det}")

    print(f"{'─'*72}")


if __name__ == "__main__":
    main()
