#!/usr/bin/env python3
"""
Benchmark script for CCTagNet models.

Measures:
  - Accuracy metrics on all test suites (detection F1, pixel IoU, center L2 error)
  - GPU latency  (CUDA events, batch=1, warmup + N runs)
  - CPU latency  (time.perf_counter, batch=1, warmup + N runs)
  - GPU throughput (FPS at larger batch size)

Example:
  python src/benchmark.py
  python src/benchmark.py --runs_dir outputs/runs --suites_dir outputs/testing \
      --output outputs/inference/benchmark_report --warmup 20 --latency_runs 100
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

# ── import model/helper code from infer script ────────────────────────────────
# Add src/ to path so we can import sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from infer_cctag_heatmap import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_EXTS,
    CombinedHeatmapLoss,
    HeatmapDiceLoss,
    decode_center_subpixel,
    infer_dataset_dir,
    load_ground_truth_map,
    load_model,
    preprocess,
    resize_heatmap_to_shape,
    safe_divide,
    summarize_evaluation,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCTagNet benchmark: accuracy + latency")
    p.add_argument("--runs_dir", type=Path, default=Path("outputs/runs"),
                   help="Directory containing model run folders (each with best.pt)")
    p.add_argument("--models", type=str, nargs="*", default=None,
                   help="Specific run names to benchmark (default: all in --runs_dir)")
    p.add_argument("--suites_dir", type=Path, default=Path("outputs/testing"),
                   help="Root directory for testing suites")
    p.add_argument("--suites", type=str, nargs="*", default=None,
                   help="Specific suite paths relative to --suites_dir (default: auto-discover)")
    p.add_argument("--output", type=Path, default=Path("outputs/inference/benchmark_report"),
                   help="Output directory for per-model JSON summaries and final report")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Heatmap peak threshold for detection")
    p.add_argument("--heatmap_binary_threshold", type=float, default=0.5,
                   help="Threshold to binarise heatmaps for pixel metrics")
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="Batch size used during accuracy evaluation (GPU)")
    p.add_argument("--warmup", type=int, default=20,
                   help="Warmup iterations before latency measurement")
    p.add_argument("--latency_runs", type=int, default=100,
                   help="Number of timed iterations for latency measurement")
    p.add_argument("--throughput_batch_size", type=int, default=16,
                   help="Batch size for GPU throughput measurement")
    p.add_argument("--latency_image", type=Path, default=None,
                   help="Single image to use for latency measurement (default: first image found)")
    p.add_argument("--cpu_latency_runs", type=int, default=30,
                   help="Number of timed iterations for CPU latency (slower)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip models whose model_summary.json already exists and suites "
                        "whose evaluation_summary.json already exists")
    return p.parse_args()


# ── latency helpers ───────────────────────────────────────────────────────────

def _latency_stats(timings: list[float], prefix: str) -> dict[str, float]:
    arr = np.array(timings)
    return {
        f"{prefix}_mean_ms":   float(arr.mean()),
        f"{prefix}_median_ms": float(np.median(arr)),
        f"{prefix}_p95_ms":    float(np.percentile(arr, 95)),
        f"{prefix}_p99_ms":    float(np.percentile(arr, 99)),
        f"{prefix}_min_ms":    float(arr.min()),
    }


def _gpu_timed_runs(model: nn.Module, x: torch.Tensor, warmup: int, runs: int,
                    use_amp: bool = False) -> list[float]:
    """Warmup then time `runs` forward passes with CUDA events. Returns ms list."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
        torch.cuda.synchronize()
        timings = []
        for _ in range(runs):
            start.record()
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    return timings


def _cpu_timed_runs(model: nn.Module, x: torch.Tensor, warmup: int, runs: int) -> list[float]:
    """Warmup then time `runs` CPU forward passes. Returns ms list."""
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def _gpu_throughput_fps(model: nn.Module, x: torch.Tensor, batch_size: int,
                        warmup: int, runs: int, use_amp: bool = False) -> float:
    batch = x.repeat(batch_size, 1, 1, 1)
    with torch.inference_mode():
        for _ in range(warmup):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(batch)
            else:
                model(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(batch)
            else:
                model(batch)
        torch.cuda.synchronize()
    return (runs * batch_size) / (time.perf_counter() - t0)


# compile warmup needs more iterations to trigger and stabilise JIT compilation
_COMPILE_WARMUP_EXTRA = 10


def measure_all_latency(
    model: nn.Module,
    probe_image_path: Path,
    config: dict,
    gpu_device: torch.device,
    cpu_device: torch.device,
    warmup: int,
    runs: int,
    cpu_runs: int,
    throughput_batch_size: int,
) -> tuple[dict[str, float], nn.Module]:
    """
    Measure all latency variants.  Returns (results_dict, model_on_gpu).

    Variants measured
    -----------------
    GPU (CUDA events, bs=1):
      gpu_baseline   – plain fp32 forward
      gpu_amp        – autocast float16
      gpu_compile    – torch.compile, fp32
      gpu_compile_amp – torch.compile + autocast float16

    GPU throughput (wall-clock, bs=N):
      gpu_tput_baseline_fps
      gpu_tput_compile_fps

    CPU (wall-clock, bs=1):
      cpu_baseline   – plain forward on CPU
      cpu_compile    – torch.compile on CPU

    Full pipeline (wall-clock, bs=1):
      gpu_pipeline   – preprocess (CPU) + H2D copy + forward + D2H + decode
                       Most relevant metric for real-time camera use.
    """
    input_w = config.get("input_width", 640)
    input_h = config.get("input_height", 400)
    results: dict[str, float] = {}

    # ── GPU variants ──────────────────────────────────────────────────────────
    model_gpu = model.to(gpu_device).eval()
    gpu_x, _ = preprocess(probe_image_path, input_w, input_h, gpu_device)

    print(f"  [gpu baseline]      ", end="", flush=True)
    t = _gpu_timed_runs(model_gpu, gpu_x, warmup, runs, use_amp=False)
    results.update(_latency_stats(t, "gpu_baseline"))
    print(f"mean={results['gpu_baseline_mean_ms']:.2f}ms  p99={results['gpu_baseline_p99_ms']:.2f}ms")

    print(f"  [gpu AMP fp16]      ", end="", flush=True)
    t = _gpu_timed_runs(model_gpu, gpu_x, warmup, runs, use_amp=True)
    results.update(_latency_stats(t, "gpu_amp"))
    print(f"mean={results['gpu_amp_mean_ms']:.2f}ms  p99={results['gpu_amp_p99_ms']:.2f}ms")

    print(f"  [gpu compile]       compiling...", end="", flush=True)
    model_gpu_compiled = torch.compile(model_gpu)
    _gpu_timed_runs(model_gpu_compiled, gpu_x, warmup + _COMPILE_WARMUP_EXTRA, runs, use_amp=False)
    t = _gpu_timed_runs(model_gpu_compiled, gpu_x, 0, runs, use_amp=False)
    results.update(_latency_stats(t, "gpu_compile"))
    print(f"\r  [gpu compile]       mean={results['gpu_compile_mean_ms']:.2f}ms  p99={results['gpu_compile_p99_ms']:.2f}ms")

    print(f"  [gpu compile+AMP]   ", end="", flush=True)
    t = _gpu_timed_runs(model_gpu_compiled, gpu_x, warmup, runs, use_amp=True)
    results.update(_latency_stats(t, "gpu_compile_amp"))
    print(f"mean={results['gpu_compile_amp_mean_ms']:.2f}ms  p99={results['gpu_compile_amp_p99_ms']:.2f}ms")

    print(f"  [gpu throughput bs={throughput_batch_size}] baseline...", end="", flush=True)
    results["gpu_tput_baseline_fps"] = _gpu_throughput_fps(
        model_gpu, gpu_x, throughput_batch_size, warmup, runs, use_amp=False)
    print(f" {results['gpu_tput_baseline_fps']:.0f} FPS  |  compile...", end="", flush=True)
    results["gpu_tput_compile_fps"] = _gpu_throughput_fps(
        model_gpu_compiled, gpu_x, throughput_batch_size, warmup, runs, use_amp=False)
    print(f" {results['gpu_tput_compile_fps']:.0f} FPS")

    # ── CPU variants ──────────────────────────────────────────────────────────
    cpu_warmup = min(warmup, 5)
    model_cpu = model.cpu().eval()
    cpu_x, _ = preprocess(probe_image_path, input_w, input_h, cpu_device)

    print(f"  [cpu baseline]      ", end="", flush=True)
    t = _cpu_timed_runs(model_cpu, cpu_x, cpu_warmup, cpu_runs)
    results.update(_latency_stats(t, "cpu_baseline"))
    print(f"mean={results['cpu_baseline_mean_ms']:.1f}ms  p99={results['cpu_baseline_p99_ms']:.1f}ms")

    print(f"  [cpu compile]       compiling...", end="", flush=True)
    model_cpu_compiled = torch.compile(model_cpu)
    _cpu_timed_runs(model_cpu_compiled, cpu_x, cpu_warmup + _COMPILE_WARMUP_EXTRA, cpu_runs)
    t = _cpu_timed_runs(model_cpu_compiled, cpu_x, 0, cpu_runs)
    results.update(_latency_stats(t, "cpu_compile"))
    print(f"\r  [cpu compile]       mean={results['cpu_compile_mean_ms']:.1f}ms  p99={results['cpu_compile_p99_ms']:.1f}ms")

    # ── Full pipeline (most relevant for real-time camera) ───────────────────
    # Measures: read+preprocess (CPU) → H2D copy → forward → D2H → peak decode
    print(f"  [gpu full pipeline] warmup...", end="", flush=True)
    model_gpu = model_gpu.to(gpu_device)  # ensure back on GPU after CPU variants
    with torch.inference_mode():
        for _ in range(warmup):
            _t, _ = preprocess(probe_image_path, input_w, input_h, cpu_device)
            _t = _t.to(gpu_device)
            _out = model_gpu(_t)
            torch.cuda.synchronize()
            decode_center_subpixel(_out[0, 0].cpu().numpy(), threshold=0.5)
        pipeline_timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _t, _ = preprocess(probe_image_path, input_w, input_h, cpu_device)
            _t = _t.to(gpu_device)
            _out = model_gpu(_t)
            torch.cuda.synchronize()
            decode_center_subpixel(_out[0, 0].cpu().numpy(), threshold=0.5)
            pipeline_timings.append((time.perf_counter() - t0) * 1000.0)
    results.update(_latency_stats(pipeline_timings, "gpu_pipeline"))
    pipeline_fps = 1000.0 / results["gpu_pipeline_mean_ms"]
    print(f"\r  [gpu full pipeline] mean={results['gpu_pipeline_mean_ms']:.2f}ms  "
          f"p99={results['gpu_pipeline_p99_ms']:.2f}ms  ({pipeline_fps:.1f} FPS theoretical max)")

    results["gpu_throughput_batch_size"] = throughput_batch_size  # keep for reference
    return results, model_gpu


# ── accuracy evaluation ───────────────────────────────────────────────────────

def evaluate_suite(
    model: nn.Module,
    suite_dir: Path,
    device: torch.device,
    config: dict,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Run accuracy evaluation on a single suite. Returns metrics dict or None if no GT."""
    images_dir = suite_dir / "images"
    if not images_dir.is_dir():
        return None

    dataset_dir = infer_dataset_dir(images_dir)
    if dataset_dir is None:
        return None  # no labels.csv / heatmaps

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        return None

    input_w = config.get("input_width", 640)
    input_h = config.get("input_height", 400)

    gt_map = load_ground_truth_map(dataset_dir)
    combined_criterion = CombinedHeatmapLoss()
    bce_criterion = nn.BCELoss()
    dice_criterion = HeatmapDiceLoss()
    eval_rows: list[dict] = []

    with torch.inference_mode():
        for batch_start in range(0, len(image_paths), args.eval_batch_size):
            batch_paths = image_paths[batch_start: batch_start + args.eval_batch_size]
            orig_sizes: dict[Path, tuple[int, int]] = {}
            tensors = []
            for p in batch_paths:
                t, orig = preprocess(p, input_w, input_h, device)
                tensors.append(t)
                orig_sizes[p] = orig
            batch_input = torch.cat(tensors, dim=0)
            batch_hm = model(batch_input)

            for i, img_path in enumerate(batch_paths):
                heatmap_tensor = batch_hm[i: i + 1].float()
                heatmap = heatmap_tensor[0, 0].cpu().numpy()
                peak_val = float(heatmap.max())
                result = decode_center_subpixel(heatmap, threshold=args.threshold)

                stem = img_path.stem
                gt_row = gt_map.get(stem)
                if gt_row is None:
                    continue

                gt_heatmap_path = gt_row["heatmap_path"]
                if not gt_heatmap_path.is_file():
                    continue

                gt_heatmap_raw = np.load(gt_heatmap_path).astype(np.float32)
                gt_heatmap = resize_heatmap_to_shape(gt_heatmap_raw, heatmap.shape)
                gt_tensor = torch.from_numpy(gt_heatmap[None, None]).to(device)

                combined_loss = float(combined_criterion(heatmap_tensor.to(device), gt_tensor).item())
                bce_loss = float(bce_criterion(heatmap_tensor.to(device), gt_tensor).item())
                dice_loss = float(dice_criterion(heatmap_tensor.to(device), gt_tensor).item())

                pred_bin = heatmap >= args.heatmap_binary_threshold
                gt_bin = gt_heatmap >= args.heatmap_binary_threshold
                tp_px = int(np.logical_and(pred_bin, gt_bin).sum())
                fp_px = int(np.logical_and(pred_bin, ~gt_bin).sum())
                fn_px = int(np.logical_and(~pred_bin, gt_bin).sum())
                tn_px = int(np.logical_and(~pred_bin, ~gt_bin).sum())

                is_neg = bool(gt_row["is_negative"])
                gt_has = not is_neg
                pred_has = result is not None

                tp_det = int(gt_has and pred_has)
                fp_det = int((not gt_has) and pred_has)
                fn_det = int(gt_has and (not pred_has))
                tn_det = int((not gt_has) and (not pred_has))

                center_l2: float | None = None
                if gt_has and pred_has:
                    orig_w, orig_h_ = orig_sizes[img_path]
                    hm_h, hm_w = heatmap.shape
                    cx_px = result[0] * orig_w / hm_w
                    cy_px = result[1] * orig_h_ / hm_h
                    center_l2 = float(np.hypot(cx_px - gt_row["center_x"], cy_px - gt_row["center_y"]))

                eval_rows.append({
                    "filename": stem,
                    "peak": peak_val,
                    "pred_detected": int(pred_has),
                    "gt_detected": int(gt_has),
                    "combined_loss": combined_loss,
                    "bce_loss": bce_loss,
                    "dice_loss": dice_loss,
                    "center_l2_px": center_l2,
                    "tp_pixels": tp_px, "fp_pixels": fp_px,
                    "fn_pixels": fn_px, "tn_pixels": tn_px,
                    "tp_det": tp_det, "fp_det": fp_det,
                    "fn_det": fn_det, "tn_det": tn_det,
                    "is_negative_gt": int(is_neg),
                    "has_visible_marker": int(gt_row["has_visible_marker"]),
                })

    if not eval_rows:
        return None

    tp_px = sum(r["tp_pixels"] for r in eval_rows)
    fp_px = sum(r["fp_pixels"] for r in eval_rows)
    fn_px = sum(r["fn_pixels"] for r in eval_rows)
    tn_px = sum(r["tn_pixels"] for r in eval_rows)
    tp_d  = sum(r["tp_det"] for r in eval_rows)
    fp_d  = sum(r["fp_det"] for r in eval_rows)
    fn_d  = sum(r["fn_det"] for r in eval_rows)
    tn_d  = sum(r["tn_det"] for r in eval_rows)

    precision = safe_divide(tp_px, tp_px + fp_px)
    recall    = safe_divide(tp_px, tp_px + fn_px)
    f1        = safe_divide(2 * precision * recall, precision + recall)
    iou       = safe_divide(tp_px, tp_px + fp_px + fn_px)
    accuracy  = safe_divide(tp_px + tn_px, tp_px + tn_px + fp_px + fn_px)

    det_prec = safe_divide(tp_d, tp_d + fp_d)
    det_rec  = safe_divide(tp_d, tp_d + fn_d)
    det_f1   = safe_divide(2 * det_prec * det_rec, det_prec + det_rec)
    det_acc  = safe_divide(tp_d + tn_d, tp_d + tn_d + fp_d + fn_d)

    center_errors = [r["center_l2_px"] for r in eval_rows if r["center_l2_px"] is not None]

    return {
        "num_images": len(eval_rows),
        "num_positive_gt": sum(0 if r["is_negative_gt"] else 1 for r in eval_rows),
        "num_negative_gt": sum(1 if r["is_negative_gt"] else 0 for r in eval_rows),
        "avg_combined_loss": float(np.mean([r["combined_loss"] for r in eval_rows])),
        "avg_bce_loss": float(np.mean([r["bce_loss"] for r in eval_rows])),
        "avg_dice_loss": float(np.mean([r["dice_loss"] for r in eval_rows])),
        "heatmap_pixel_accuracy": accuracy,
        "heatmap_pixel_precision": precision,
        "heatmap_pixel_recall": recall,
        "heatmap_pixel_f1": f1,
        "heatmap_pixel_iou": iou,
        "detection_accuracy": det_acc,
        "detection_precision": det_prec,
        "detection_recall": det_rec,
        "detection_f1": det_f1,
        "mean_center_l2_px": float(np.mean(center_errors)) if center_errors else None,
        "rows": eval_rows,
    }


# ── discover suites ───────────────────────────────────────────────────────────

def discover_suites(suites_dir: Path) -> list[Path]:
    """Find all directories under suites_dir that contain images/ and labels.csv."""
    found = []
    for candidate in sorted(suites_dir.rglob("images")):
        parent = candidate.parent
        if (parent / "labels.csv").is_file() and (parent / "heatmaps").is_dir():
            found.append(parent)
    return found


# ── print helpers ─────────────────────────────────────────────────────────────

def print_table(headers: list[str], rows: list[list[str]], col_width: int = 18) -> None:
    fmt = " | ".join(f"{{:<{col_width}}}" for _ in headers)
    sep = "-+-".join("-" * col_width for _ in headers)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    gpu_device = torch.device("cuda") if cuda_available else None
    cpu_device = torch.device("cpu")

    # ── discover model runs ──────────────────────────────────────────────────
    if args.models:
        run_dirs = [args.runs_dir / m for m in args.models]
    else:
        run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir() and (d / "best.pt").is_file())

    if not run_dirs:
        raise SystemExit(f"No model runs found in {args.runs_dir}")

    # ── discover test suites ─────────────────────────────────────────────────
    if args.suites:
        suite_dirs = [args.suites_dir / s for s in args.suites]
    else:
        suite_dirs = discover_suites(args.suites_dir)

    if not suite_dirs:
        raise SystemExit(f"No evaluable suites found under {args.suites_dir}")

    print(f"Models  : {[d.name for d in run_dirs]}")
    print(f"Suites  : {[d.name for d in suite_dirs]}")
    print(f"Output  : {args.output}")
    print(f"CUDA    : {cuda_available}")
    print()

    # ── pick latency sample image ────────────────────────────────────────────
    latency_image_path = args.latency_image
    if latency_image_path is None:
        for suite in suite_dirs:
            imgs = sorted((suite / "images").glob("*"))
            candidates = [p for p in imgs if p.suffix.lower() in IMAGE_EXTS]
            if candidates:
                latency_image_path = candidates[0]
                break
    if latency_image_path is None:
        raise SystemExit("Could not find any image for latency measurement")
    print(f"Latency probe image: {latency_image_path}\n")

    all_results: dict[str, Any] = {}

    for run_dir in run_dirs:
        model_name = run_dir.name
        checkpoint = run_dir / "best.pt"
        print(f"{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        # ── latency: use cache or measure ────────────────────────────────────
        model_json = args.output / model_name / "model_summary.json"
        _LATENCY_SENTINEL = "gpu_baseline_mean_ms"  # absent in old-format caches

        _use_latency_cache = (
            args.skip_existing
            and model_json.is_file()
            and _LATENCY_SENTINEL in json.loads(model_json.read_text(encoding="utf-8"))
        )

        if _use_latency_cache:
            cached = json.loads(model_json.read_text(encoding="utf-8"))
            latency_results = {k: v for k, v in cached.items()
                               if k not in ("model", "backbone", "checkpoint", "suites")}
            backbone = cached.get("backbone", "?")
            config: dict = {}
            model = None
            print(f"  backbone={backbone}  [latency cached]")
        else:
            # ── load model ───────────────────────────────────────────────────
            device = gpu_device if cuda_available else cpu_device
            try:
                model, config = load_model(checkpoint, device)
            except Exception as exc:
                print(f"  [SKIP] failed to load checkpoint: {exc}")
                continue
            backbone = config.get("backbone", "efficientnet_b0")
            input_w  = config.get("input_width", 640)
            input_h  = config.get("input_height", 400)
            print(f"  backbone={backbone}  input={input_w}x{input_h}  "
                  f"warmup={args.warmup}  runs={args.latency_runs}")

            # ── measure all latency variants ─────────────────────────────────
            if cuda_available:
                latency_results, model = measure_all_latency(
                    model, latency_image_path, config,
                    gpu_device, cpu_device,
                    warmup=args.warmup,
                    runs=args.latency_runs,
                    cpu_runs=args.cpu_latency_runs,
                    throughput_batch_size=args.throughput_batch_size,
                )
            else:
                model_cpu = model.cpu().eval()
                cpu_x, _ = preprocess(latency_image_path, input_w, input_h, cpu_device)
                t = _cpu_timed_runs(model_cpu, cpu_x, min(args.warmup, 5), args.cpu_latency_runs)
                latency_results = _latency_stats(t, "cpu_baseline")
                model = model_cpu

        # ── accuracy evaluation per suite ────────────────────────────────────
        # load cached suite results if skip_existing
        cached_suites: dict[str, Any] = {}
        if args.skip_existing and model_json.is_file():
            cached_suites = json.loads(model_json.read_text(encoding="utf-8")).get("suites", {})

        suite_metrics: dict[str, Any] = {}
        for suite_dir in suite_dirs:
            suite_json = args.output / model_name / suite_dir.name / "evaluation_summary.json"

            if args.skip_existing and suite_json.is_file():
                suite_metrics[suite_dir.name] = json.loads(suite_json.read_text(encoding="utf-8"))
                print(f"  [eval] {suite_dir.name} ... cached")
                continue

            # need the model loaded to run eval
            if model is None:
                device = gpu_device if cuda_available else cpu_device
                try:
                    model, config = load_model(checkpoint, device)
                except Exception as exc:
                    print(f"  [eval] {suite_dir.name} ... SKIP (cannot load model: {exc})")
                    continue
                if cuda_available:
                    model = model.to(gpu_device)

            print(f"  [eval] {suite_dir.name} ...", end=" ", flush=True)
            metrics = evaluate_suite(model, suite_dir, device, config, args)
            if metrics is None:
                print("skipped (no GT)")
                continue
            rows = metrics.pop("rows")
            suite_metrics[suite_dir.name] = metrics

            suite_out = args.output / model_name / suite_dir.name
            suite_out.mkdir(parents=True, exist_ok=True)
            csv_path = suite_out / "evaluation_per_image.csv"
            if rows:
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            suite_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            print(f"det_f1={metrics['detection_f1']:.4f}  "
                  f"iou={metrics['heatmap_pixel_iou']:.4f}  "
                  f"center_l2={metrics['mean_center_l2_px']:.2f}px"
                  if metrics['mean_center_l2_px'] is not None else
                  f"det_f1={metrics['detection_f1']:.4f}  iou={metrics['heatmap_pixel_iou']:.4f}")

        model_result = {
            "model": model_name,
            "backbone": backbone,
            "checkpoint": str(checkpoint),
            **latency_results,
            "suites": suite_metrics,
        }
        all_results[model_name] = model_result

        model_json.parent.mkdir(parents=True, exist_ok=True)
        model_json.write_text(json.dumps(model_result, indent=2), encoding="utf-8")
        print()

    # ── save full report ─────────────────────────────────────────────────────
    report_path = args.output / "benchmark_report.json"
    report_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # ── print summary tables ─────────────────────────────────────────────────
    model_names = list(all_results.keys())
    all_suites = sorted({s for r in all_results.values() for s in r["suites"]})

    def _g(res: dict, key: str, fmt: str = ".2f") -> str:
        v = res.get(key)
        return f"{v:{fmt}}" if v is not None else "n/a"

    print(f"\n{'='*80}")
    print("GPU LATENCY (bs=1, CUDA events)  —  forward pass only")
    print(f"{'='*80}")
    print_table(
        ["model", "baseline_ms", "AMP_ms", "compile_ms", "compile+AMP_ms", "pipeline_ms", "pipeline_fps"],
        [[mn,
          _g(r, "gpu_baseline_mean_ms"),
          _g(r, "gpu_amp_mean_ms"),
          _g(r, "gpu_compile_mean_ms"),
          _g(r, "gpu_compile_amp_mean_ms"),
          _g(r, "gpu_pipeline_mean_ms"),
          f"{1000/r['gpu_pipeline_mean_ms']:.1f}" if r.get("gpu_pipeline_mean_ms") else "n/a",
         ] for mn, r in all_results.items()],
        col_width=18,
    )

    print(f"\n{'='*80}")
    print("GPU THROUGHPUT (bs=16, wall-clock FPS)")
    print(f"{'='*80}")
    print_table(
        ["model", "baseline_fps", "compile_fps"],
        [[mn, _g(r, "gpu_tput_baseline_fps", ".0f"), _g(r, "gpu_tput_compile_fps", ".0f")]
         for mn, r in all_results.items()],
        col_width=22,
    )

    print(f"\n{'='*80}")
    print("CPU LATENCY (bs=1, wall-clock)")
    print(f"{'='*80}")
    print_table(
        ["model", "baseline_ms", "compile_ms"],
        [[mn, _g(r, "cpu_baseline_mean_ms", ".1f"), _g(r, "cpu_compile_mean_ms", ".1f")]
         for mn, r in all_results.items()],
        col_width=22,
    )

    print(f"\n{'='*60}")
    print("DETECTION F1 BY SUITE")
    print(f"{'='*60}")
    print_table(["suite"] + model_names,
                [[s] + [f"{all_results[mn]['suites'].get(s, {}).get('detection_f1', float('nan')):.4f}"
                        for mn in model_names]
                 for s in all_suites],
                col_width=24)

    print(f"\n{'='*60}")
    print("MEAN CENTER L2 ERROR (px) BY SUITE")
    print(f"{'='*60}")
    print_table(["suite"] + model_names,
                [[s] + [f"{all_results[mn]['suites'].get(s, {}).get('mean_center_l2_px') or float('nan'):.2f}"
                        for mn in model_names]
                 for s in all_suites],
                col_width=24)

    print(f"\n{'='*60}")
    print("HEATMAP IoU BY SUITE")
    print(f"{'='*60}")
    print_table(["suite"] + model_names,
                [[s] + [f"{all_results[mn]['suites'].get(s, {}).get('heatmap_pixel_iou', float('nan')):.4f}"
                        for mn in model_names]
                 for s in all_suites],
                col_width=24)

    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
