#!/usr/bin/env python3
"""
Inference script for CCTagNet heatmap model.

Example:
  # single image
  python infer_cctag_heatmap.py --checkpoint ./runs/experiment_01/best.pt --input image.png

  # directory of images (saves heatmaps and prints centers)
  python infer_cctag_heatmap.py --checkpoint ./runs/experiment_01/best.pt --input ./images/ --output ./results/
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path, PosixPath
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCTagNet inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint (best.pt / last.pt)")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory of images")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for heatmap .npy and overlay .png (optional)")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0 (default: auto)")
    parser.add_argument("--vis", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap peak threshold for detection (default: 0.5)")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Dataset root containing images/, heatmaps/, and labels.csv for evaluation. "
             "If omitted and --input is DATASET/images, infer automatically.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate against dataset ground truth when heatmaps/ and labels.csv are available.",
    )
    parser.add_argument(
        "--heatmap_binary_threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize predicted/GT heatmaps for precision, recall, F1, and IoU.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── model definition (must match training) ────────────────────────────────────

def ensure_torchvision():
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise SystemExit("pip install torchvision") from exc
    return efficientnet_b0, EfficientNet_B0_Weights


class CCTagNet(nn.Module):
    def __init__(self, heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        efficientnet_b0, EfficientNet_B0_Weights = ensure_torchvision()
        backbone = efficientnet_b0(weights=None)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1280, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128,  64, kernel_size=3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d( 64,  32, kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pred = self.decoder(features)
        return nn.functional.interpolate(
            pred, size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear", align_corners=False,
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CCTagNet, dict]:
    try:
        with torch.serialization.safe_globals([PosixPath]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # PyTorch 2.6 defaults to weights_only=True, but older checkpoints may
        # embed trusted Python objects such as pathlib.PosixPath in metadata.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    heatmap_size = (config.get("heatmap_height", 100), config.get("heatmap_width", 160))
    model = CCTagNet(heatmap_size=heatmap_size).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def preprocess(image_path: Path, input_width: int, input_height: int, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0).to(device), (orig_w, orig_h)


def decode_center(heatmap: np.ndarray, threshold: float = 0.3) -> tuple[float, float] | None:
    """Return (x, y) of the peak in heatmap coords, or None if peak < threshold."""
    peak_val = float(heatmap.max())
    if peak_val < threshold:
        return None
    idx = np.argmax(heatmap)
    h, w = heatmap.shape
    y, x = divmod(int(idx), w)
    return float(x), float(y)


def decode_center_subpixel(heatmap: np.ndarray, threshold: float = 0.3) -> tuple[float, float] | None:
    peak_val = float(heatmap.max())
    if peak_val < threshold:
        return None

    hm = heatmap.astype(np.float32, copy=False)
    height, width = hm.shape
    flat_idx = int(np.argmax(hm))
    py, px = divmod(flat_idx, width)

    padded = np.pad(hm, ((1, 1), (1, 1)), mode="edge")
    px_p = px + 1
    py_p = py + 1

    p11 = float(padded[py_p, px_p])
    p10 = float(padded[py_p, px_p - 1])
    p12 = float(padded[py_p, px_p + 1])
    p01 = float(padded[py_p - 1, px_p])
    p21 = float(padded[py_p + 1, px_p])

    dx = 0.5 * (p12 - p10)
    dy = 0.5 * (p21 - p01)
    dxx = p12 - 2.0 * p11 + p10
    dyy = p21 - 2.0 * p11 + p01
    dxy = 0.25 * (
        float(padded[py_p + 1, px_p + 1])
        - float(padded[py_p + 1, px_p - 1])
        - float(padded[py_p - 1, px_p + 1])
        + float(padded[py_p - 1, px_p - 1])
    )

    det = dxx * dyy - dxy * dxy
    delta_x = 0.0
    delta_y = 0.0
    if abs(det) > 1e-6:
        delta_x = -(dyy * dx - dxy * dy) / det
        delta_y = -(-dxy * dx + dxx * dy) / det
        delta_x = float(np.clip(delta_x, -0.5, 0.5))
        delta_y = float(np.clip(delta_y, -0.5, 0.5))

    return float(px) + delta_x, float(py) + delta_y


class HeatmapDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.flatten(1)
        target = target.flatten(1)
        intersection = (prediction * target).sum(dim=1)
        denominator = prediction.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedHeatmapLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = HeatmapDiceLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(prediction, target) + 0.5 * self.dice(prediction, target)


def infer_dataset_dir(input_path: Path) -> Path | None:
    if input_path.is_dir() and input_path.name == "images":
        candidate = input_path.parent
        if (candidate / "labels.csv").is_file() and (candidate / "heatmaps").is_dir():
            return candidate
    return None


def load_ground_truth_map(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    labels_csv = dataset_dir / "labels.csv"
    heatmaps_dir = dataset_dir / "heatmaps"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing labels.csv: {labels_csv}")
    if not heatmaps_dir.is_dir():
        raise FileNotFoundError(f"Missing heatmaps directory: {heatmaps_dir}")

    gt_map: dict[str, dict[str, Any]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            gt_map[filename] = {
                "heatmap_path": heatmaps_dir / f"{filename}.npy",
                "center_x": float(row.get("center_x") or row.get("x") or -1.0),
                "center_y": float(row.get("center_y") or row.get("y") or -1.0),
                "is_negative": int(row.get("is_negative") or 0) == 1,
                "has_visible_marker": int(row.get("has_visible_marker") or 0) == 1,
            }
    return gt_map


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def resize_heatmap_to_shape(heatmap: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape_hw
    if heatmap.shape == (target_h, target_w):
        return heatmap.astype(np.float32, copy=False)
    resized = cv2.resize(heatmap.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def summarize_evaluation(
    rows: list[dict[str, Any]],
    output_dir: Path | None,
) -> None:
    if not rows:
        print("evaluation: no matching ground-truth rows found")
        return

    metrics = {
        "num_images": len(rows),
        "num_positive_gt": int(sum(0 if row["is_negative_gt"] else 1 for row in rows)),
        "num_negative_gt": int(sum(1 if row["is_negative_gt"] else 0 for row in rows)),
        "avg_combined_loss": float(np.mean([row["combined_loss"] for row in rows])),
        "avg_bce_loss": float(np.mean([row["bce_loss"] for row in rows])),
        "avg_dice_loss": float(np.mean([row["dice_loss"] for row in rows])),
    }

    tp_px = float(sum(row["tp_pixels"] for row in rows))
    fp_px = float(sum(row["fp_pixels"] for row in rows))
    fn_px = float(sum(row["fn_pixels"] for row in rows))
    tn_px = float(sum(row["tn_pixels"] for row in rows))

    precision = safe_divide(tp_px, tp_px + fp_px)
    recall = safe_divide(tp_px, tp_px + fn_px)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    iou = safe_divide(tp_px, tp_px + fp_px + fn_px)
    accuracy = safe_divide(tp_px + tn_px, tp_px + tn_px + fp_px + fn_px)

    tp_det = float(sum(row["tp_det"] for row in rows))
    fp_det = float(sum(row["fp_det"] for row in rows))
    fn_det = float(sum(row["fn_det"] for row in rows))
    tn_det = float(sum(row["tn_det"] for row in rows))

    det_precision = safe_divide(tp_det, tp_det + fp_det)
    det_recall = safe_divide(tp_det, tp_det + fn_det)
    det_f1 = safe_divide(2.0 * det_precision * det_recall, det_precision + det_recall)
    det_accuracy = safe_divide(tp_det + tn_det, tp_det + tn_det + fp_det + fn_det)

    center_errors = [row["center_l2_px"] for row in rows if row["center_l2_px"] is not None]
    metrics.update(
        {
            "heatmap_pixel_accuracy": accuracy,
            "heatmap_pixel_precision": precision,
            "heatmap_pixel_recall": recall,
            "heatmap_pixel_f1": f1,
            "heatmap_pixel_iou": iou,
            "detection_accuracy": det_accuracy,
            "detection_precision": det_precision,
            "detection_recall": det_recall,
            "detection_f1": det_f1,
            "mean_center_l2_px_on_detected_positives": float(np.mean(center_errors)) if center_errors else None,
        }
    )

    print("\n=== evaluation summary ===")
    for key, value in metrics.items():
        if value is None:
            print(f"{key}: n/a")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "evaluation_summary.json"
        per_image_path = output_dir / "evaluation_per_image.csv"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        fieldnames = list(rows[0].keys())
        with per_image_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved evaluation summary: {summary_path}")
        print(f"saved per-image metrics: {per_image_path}")


def save_overlay(image_path: Path, heatmap: np.ndarray, center_xy: tuple[float, float], out_path: Path) -> None:
    orig = cv2.imread(str(image_path))
    h, w = orig.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_u8 = (hm_resized * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.6, hm_color, 0.4, 0)

    # scale center from heatmap coords → original image coords
    hm_h, hm_w = heatmap.shape
    cx = int(center_xy[0] * w / hm_w)
    cy = int(center_xy[1] * h / hm_h)
    cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)
    cv2.circle(overlay, (cx, cy), 8, (0, 0, 0),   2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    input_width  = config.get("input_width",  640)
    input_height = config.get("input_height", 400)
    print(f"model input: {input_width}x{input_height}  device: {device}")

    # collect image paths
    if args.input.is_dir():
        image_paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        image_paths = [args.input]

    if not image_paths:
        raise SystemExit(f"No images found at {args.input}")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    dataset_dir = args.dataset_dir
    if dataset_dir is None and args.eval:
        dataset_dir = infer_dataset_dir(args.input)

    gt_map: dict[str, dict[str, Any]] | None = None
    if args.eval:
        if dataset_dir is None:
            raise SystemExit(
                "--eval requires --dataset_dir, or --input must be DATASET/images with sibling labels.csv and heatmaps/"
            )
        print(f"evaluation dataset: {dataset_dir}")
        gt_map = load_ground_truth_map(dataset_dir)
        combined_criterion = CombinedHeatmapLoss()
        bce_criterion = nn.BCELoss()
        dice_criterion = HeatmapDiceLoss()
        eval_rows: list[dict[str, Any]] = []
    else:
        combined_criterion = None
        bce_criterion = None
        dice_criterion = None
        eval_rows = []

    with torch.no_grad():
        for img_path in image_paths:
            tensor, (orig_w, orig_h) = preprocess(img_path, input_width, input_height, device)
            heatmap_tensor = model(tensor)                        # (1, 1, H, W)
            heatmap = heatmap_tensor[0, 0].cpu().numpy()         # (H, W)

            peak_val = float(heatmap.max())
            result = decode_center_subpixel(heatmap, threshold=args.threshold)

            if result is None:
                print(f"{img_path.name}  NO DETECTION  peak={peak_val:.3f} (< {args.threshold})")
            else:
                cx_hm, cy_hm = result
                hm_h, hm_w = heatmap.shape
                cx_px = cx_hm * orig_w / hm_w
                cy_px = cy_hm * orig_h / hm_h
                print(f"{img_path.name}  center=({cx_px:.1f}, {cy_px:.1f})px  heatmap_peak=({cx_hm:.1f}, {cy_hm:.1f})  peak={peak_val:.3f}")

            if args.output:
                stem = img_path.stem
                np.save(args.output / f"{stem}_heatmap.npy", heatmap)
                if args.vis and result is not None:
                    save_overlay(img_path, heatmap, result, args.output / f"{stem}_overlay.png")

            if gt_map is not None:
                stem = img_path.stem
                gt_row = gt_map.get(stem)
                if gt_row is None:
                    print(f"{img_path.name}  warning: missing GT row in labels.csv, skipped evaluation")
                    continue

                gt_heatmap_path = gt_row["heatmap_path"]
                if not gt_heatmap_path.is_file():
                    print(f"{img_path.name}  warning: missing GT heatmap {gt_heatmap_path}, skipped evaluation")
                    continue

                gt_heatmap_raw = np.load(gt_heatmap_path).astype(np.float32)
                gt_heatmap = resize_heatmap_to_shape(gt_heatmap_raw, heatmap.shape)
                gt_tensor = torch.from_numpy(gt_heatmap[None, None, ...]).to(device)

                combined_loss = float(combined_criterion(heatmap_tensor, gt_tensor).cpu().item())
                bce_loss = float(bce_criterion(heatmap_tensor, gt_tensor).cpu().item())
                dice_loss = float(dice_criterion(heatmap_tensor, gt_tensor).cpu().item())

                pred_bin = heatmap >= args.heatmap_binary_threshold
                gt_bin = gt_heatmap >= args.heatmap_binary_threshold
                tp_pixels = int(np.logical_and(pred_bin, gt_bin).sum())
                fp_pixels = int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
                fn_pixels = int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())
                tn_pixels = int(np.logical_and(np.logical_not(pred_bin), np.logical_not(gt_bin)).sum())

                is_negative_gt = bool(gt_row["is_negative"])
                gt_has_object = not is_negative_gt
                pred_has_object = result is not None

                tp_det = int(gt_has_object and pred_has_object)
                fp_det = int((not gt_has_object) and pred_has_object)
                fn_det = int(gt_has_object and (not pred_has_object))
                tn_det = int((not gt_has_object) and (not pred_has_object))

                center_l2_px: float | None = None
                if gt_has_object and pred_has_object:
                    gt_cx = float(gt_row["center_x"])
                    gt_cy = float(gt_row["center_y"])
                    center_l2_px = float(np.hypot(cx_px - gt_cx, cy_px - gt_cy))

                eval_rows.append(
                    {
                        "filename": stem,
                        "peak": peak_val,
                        "pred_detected": int(pred_has_object),
                        "gt_detected": int(gt_has_object),
                        "combined_loss": combined_loss,
                        "bce_loss": bce_loss,
                        "dice_loss": dice_loss,
                        "center_l2_px": center_l2_px,
                        "tp_pixels": tp_pixels,
                        "fp_pixels": fp_pixels,
                        "fn_pixels": fn_pixels,
                        "tn_pixels": tn_pixels,
                        "tp_det": tp_det,
                        "fp_det": fp_det,
                        "fn_det": fn_det,
                        "tn_det": tn_det,
                        "is_negative_gt": int(is_negative_gt),
                        "has_visible_marker": int(gt_row["has_visible_marker"]),
                        "gt_heatmap_height_raw": int(gt_heatmap_raw.shape[0]),
                        "gt_heatmap_width_raw": int(gt_heatmap_raw.shape[1]),
                        "eval_heatmap_height": int(gt_heatmap.shape[0]),
                        "eval_heatmap_width": int(gt_heatmap.shape[1]),
                    }
                )

    if gt_map is not None:
        summarize_evaluation(eval_rows, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
