#!/usr/bin/env python3
"""
Fine-tune a YOLO-nano detector for CCTag acquisition (stage 1 of the two-stage pipeline).

This is a thin wrapper over Ultralytics. The detector only has to answer
"is there a CCTag and roughly where + how big" across a wide range of apparent
sizes -- precision is handled downstream by the heatmap localization model, so a
nano backbone is the right tier (small + fast for the multi-scale scan).

Requires the `detect` extra:  uv sync --extra cu126 --extra detect

Example:
  uv run python src/train_yolo_detection.py \
    --data ./outputs/datasets/yolo_detection/data.yaml \
    --model yolo11n.pt --epochs 100 --imgsz 640 --batch 64 \
    --project ./outputs/runs_yolo --name cctag_det_n
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO-nano CCTag detector (Ultralytics)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to data.yaml (from prepare_yolo_dataset.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base weights to fine-tune (e.g. yolo11n.pt, yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training image size (square; letterboxed). Keep equal to the deploy "
        "imgsz: 2x2 tiling of 4096 -> ~2048 tiles resized to 1024.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=20,
        help="Batch size (-1 lets Ultralytics auto-pick).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA index/list or 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("./outputs/runs_yolo"),
        help="Run output root.",
    )
    parser.add_argument("--name", type=str, default="cctag_det_n", help="Run name.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (per GPU process).")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=["ram", "disk"],
        help="Cache images to remove the disk-decode bottleneck (nano models are dataloader-bound). "
        "'ram' keeps decoded images in memory (needs enough RAM); 'disk' caches preprocessed *.npy.",
    )
    # Scale-robustness augmentation: the detector must fire across a wide apparent-size
    # band so acquisition needs fewer external scales. These widen the effective band.
    parser.add_argument(
        "--scale",
        type=float,
        default=0.9,
        help="Random scale augmentation gain (Ultralytics 'scale'). Higher = wider size range.",
    )
    parser.add_argument(
        "--mosaic", type=float, default=1.0, help="Mosaic augmentation probability."
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=10,
        help="Disable mosaic for the last N epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "ultralytics is required for YOLO detection training.\n"
            "Install it with:  uv sync --extra cu126 --extra detect"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        cache=args.cache if args.cache is not None else False,
        scale=args.scale,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
    )
    print(f"done. weights under: {args.project / args.name / 'weights'}")


if __name__ == "__main__":
    main()
