#!/usr/bin/env python3
"""Render a 3-panel "triptych" figure for the report (sec. 4.3.2 network architecture):

    (a) input image  ->  (b) predicted heatmap (bright center blob)  ->  (c) decoded center

Reuses the trained model loader, preprocessing, and decoders from
``infer_cctag_heatmap.py`` so the heatmap and the marked center match exactly what
inference produces. Output is a single high-DPI PNG suitable for dropping into a doc.

Example:
    uv run python src/make_heatmap_triptych.py \
        --checkpoint outputs/runs/fable_occ/best.pt \
        --image outputs/datasets/6f_labeled_1024x640_roi/images/frame_20260606_221631_pos2.png \
        --output outputs/figures/heatmap_triptych.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# Reuse the exact inference building blocks (script dir is on sys.path when run directly).
from infer_cctag_heatmap import (
    load_model,
    preprocess,
    decode_center_weighted,
    decode_center_subpixel,
    decode_center,
    decode_center_offset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render input->heatmap->center triptych figure")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/runs/fable_occ/best.pt"),
                   help="Trained checkpoint (.pt)")
    p.add_argument("--image", type=Path, required=True, help="Input image to visualize")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/figures/heatmap_triptych.png"),
                   help="Output figure path (.png)")
    p.add_argument("--device", type=str, default=None, help="cpu / cuda (auto if unset)")
    p.add_argument("--threshold", type=float, default=0.3, help="Peak accept threshold")
    p.add_argument("--decode_method", type=str, default="offset",
                   choices=["offset", "weighted", "subpixel", "argmax"],
                   help="Center decoder; 'offset' matches production when an offset head exists")
    p.add_argument("--colormap", type=str, default="turbo",
                   help="Matplotlib colormap for the heatmap (e.g. turbo, jet, inferno)")
    p.add_argument("--overlay_alpha", type=float, default=0.45,
                   help="Heatmap transparency over the image in panel (c)")
    p.add_argument("--hm_gamma", type=float, default=0.6,
                   help="Display-only gamma for the heatmap (<1 brightens the Gaussian "
                        "falloff so the blob is visible; 1.0 = raw linear). Does NOT affect decoding.")
    p.add_argument("--zoom", type=int, default=None,
                   help="Crop a square window of this many input-space pixels centered on "
                        "the detected center, applied to all panels (close-up for the report)")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    p.add_argument("--titles", dest="titles", action="store_true", default=True,
                   help="Draw panel titles (default on)")
    p.add_argument("--no-titles", dest="titles", action="store_false",
                   help="Omit titles for a clean figure to caption in the report")
    return p.parse_args()


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) normalized tensor -> HxWx3 uint8 RGB exactly as the net sees it."""
    img = tensor[0].detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = (img.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return img


def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model(args.checkpoint, device)
    in_w = config.get("input_width", 640)
    in_h = config.get("input_height", 400)
    hm_w = config.get("heatmap_width", 256)
    hm_h = config.get("heatmap_height", 160)

    # Forward pass (fp32: fp16 plateaus the sigmoid peak and biases the center).
    tensor, orig_size = preprocess(args.image, in_w, in_h, device)
    with torch.no_grad():
        out = model(tensor)
    offset_np = None
    if isinstance(out, tuple):
        heatmap_t = out[0]
        if len(out) >= 2:
            offset_np = out[1][0].float().cpu().numpy()  # (2, hm_h, hm_w)
    else:
        heatmap_t = out
    heatmap = heatmap_t[0, 0].float().cpu().numpy()  # (hm_h, hm_w) in [0,1]

    # Decode the center in heatmap space, then scale to input-image pixels.
    if args.decode_method == "offset" and offset_np is not None:
        res = decode_center_offset(heatmap, offset_np, threshold=args.threshold)
    elif args.decode_method == "weighted":
        res = decode_center_weighted(heatmap, threshold=args.threshold)
    elif args.decode_method == "subpixel":
        res = decode_center_subpixel(heatmap, threshold=args.threshold)
    else:
        res = decode_center(heatmap, threshold=args.threshold)

    peak_val = float(heatmap.max())
    sx, sy = in_w / hm_w, in_h / hm_h
    center_in = None
    if res is not None:
        center_in = (res[0] * sx, res[1] * sy)

    # The RGB the network actually consumed (resized to in_w x in_h) -> perfect alignment.
    img_in = denormalize(tensor)
    # Smooth heatmap upsampled to input size for display/overlay.
    hm_up = cv2.resize(heatmap, (in_w, in_h), interpolation=cv2.INTER_CUBIC)
    hm_up = np.clip(hm_up, 0.0, 1.0)

    # Display-only gamma stretch so the (sharp) Gaussian falloff is visible.
    norm = PowerNorm(gamma=args.hm_gamma, vmin=0.0, vmax=1.0)

    # Optional close-up crop centered on the detection, applied to every panel.
    x0 = y0 = 0
    if args.zoom and center_in is not None:
        half = args.zoom // 2
        cx, cy = center_in
        x0 = int(np.clip(round(cx - half), 0, max(0, in_w - args.zoom)))
        y0 = int(np.clip(round(cy - half), 0, max(0, in_h - args.zoom)))
        x1, y1 = x0 + args.zoom, y0 + args.zoom
        img_in = img_in[y0:y1, x0:x1]
        hm_up = hm_up[y0:y1, x0:x1]

    # ── Figure: 3 panels ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), dpi=args.dpi)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    # (a) input
    axes[0].imshow(img_in)
    if args.titles:
        axes[0].set_title("(a) Input image", fontsize=12)

    # (b) heatmap
    im = axes[1].imshow(hm_up, cmap=args.colormap, norm=norm)
    if args.titles:
        axes[1].set_title(f"(b) Predicted heatmap  (peak={peak_val:.2f})", fontsize=12)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("confidence", fontsize=9)

    # (c) image + heatmap overlay + decoded center (subtract crop origin for alignment)
    axes[2].imshow(img_in)
    axes[2].imshow(hm_up, cmap=args.colormap, norm=norm, alpha=args.overlay_alpha)
    if center_in is not None:
        cx, cy = center_in[0] - x0, center_in[1] - y0
        axes[2].plot(cx, cy, marker="+", color="white", markersize=16,
                     markeredgewidth=2.5)
        axes[2].add_patch(plt.Circle((cx, cy), radius=10, fill=False,
                                     edgecolor="red", linewidth=2.0))
        sub = f"(c) Detected center  ({center_in[0]:.1f}, {center_in[1]:.1f})"
    else:
        sub = "(c) No detection"
    if args.titles:
        axes[2].set_title(sub, fontsize=12)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"input image      : {args.image}")
    print(f"original size    : {orig_size[0]}x{orig_size[1]}  -> net input {in_w}x{in_h}")
    print(f"heatmap          : {hm_w}x{hm_h}  peak={peak_val:.3f}  thr={args.threshold}")
    if center_in is not None:
        print(f"decoded center   : ({center_in[0]:.1f}, {center_in[1]:.1f}) px "
              f"(input space) via '{args.decode_method}'")
    else:
        print("decoded center   : NONE (peak below threshold)")
    print(f"saved figure     : {args.output}")


if __name__ == "__main__":
    main()
