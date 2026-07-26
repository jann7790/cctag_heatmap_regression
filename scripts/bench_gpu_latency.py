"""Pure GPU forward-pass latency benchmark for CCTagNet checkpoints.

Reuses the model definitions in src/infer_cctag_heatmap.py, builds the exact
backbone recorded in the checkpoint config (incl. hrnet_w18_small_v2, which
load_model's branch logic does not cover), then times the forward pass with
warmup + torch.cuda.synchronize. Reports mean / median / p95 / FPS.

Run with CUDA_VISIBLE_DEVICES=<n> to pin a GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import infer_cctag_heatmap as infer  # noqa: E402


def build_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    infer.DECODE_ALIGN_CORNERS = bool(config.get("align_corners", False))
    heatmap_size = (config.get("heatmap_height", 160), config.get("heatmap_width", 256))
    backbone = config.get("backbone", "efficientnet_b0")
    state = ckpt["model_state_dict"]
    head_kwargs = {
        "use_offset_head": bool(config.get("use_offset_head", False)),
        "use_size_head": bool(config.get("use_size_head", False)),
        "offset_head_hidden": int(config.get("offset_head_hidden", 0)),
        "decoder_blocks": int(config.get("decoder_blocks", 1)),
    }
    if backbone.startswith("hrnet"):
        model = infer.CCTagNetHRNet(
            heatmap_size=heatmap_size, variant=backbone, pretrained=False, **head_kwargs
        ).to(device)
    else:
        model, _ = infer.load_model(ckpt_path, device)
        return model, config
    model.load_state_dict(state)
    model.eval()
    return model, config


def bench(model, config, device, iters: int, warmup: int):
    h = config.get("input_height", 640)
    w = config.get("input_width", 1024)
    x = torch.randn(1, 3, h, w, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times, (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    model, config = build_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    times, (h, w) = bench(model, config, device, args.iters, args.warmup)

    times.sort()
    mean = statistics.mean(times)
    median = statistics.median(times)
    p95 = times[int(0.95 * len(times)) - 1]
    p99 = times[int(0.99 * len(times)) - 1]

    print(f"checkpoint : {args.checkpoint}")
    print(f"backbone   : {config.get('backbone')}")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"input      : 1x3x{h}x{w}")
    print(f"params     : {n_params/1e6:.2f} M")
    print(f"iters      : {len(times)} (warmup {args.warmup})")
    print(f"mean       : {mean:.3f} ms")
    print(f"median     : {median:.3f} ms")
    print(f"p95        : {p95:.3f} ms")
    print(f"p99        : {p99:.3f} ms")
    print(f"min        : {times[0]:.3f} ms")
    print(f"FPS (mean) : {1000.0/mean:.1f}")


if __name__ == "__main__":
    main()
