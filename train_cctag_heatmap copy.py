# ● 不完全是。比較一下：

#   ┌──────────┬────────────────────────────┬──────────────────────┐
#   │          │   decode_heatmap_centers   │    training loss     │
#   ├──────────┼────────────────────────────┼──────────────────────┤
#   │ 原始版本 │ argmax（整數像素）         │ heatmap only         │
#   ├──────────┼────────────────────────────┼──────────────────────┤
#   │ 現在     │ Taylor expansion（亞像素） │ heatmap + coord_loss │
#   ├──────────┼────────────────────────────┼──────────────────────┤
#   │ 選項 A   │ Taylor expansion（亞像素） │ heatmap only         │
#   └──────────┴────────────────────────────┴──────────────────────┘

#   選項 A = 保留 Taylor expansion 讓驗證指標更準，但拿掉 coord_loss。是「比原始好，但比現在簡單」的版本。

#   要回到選項 A 的話，只需要把 train_one_epoch 裡的 coord_loss 部分拿掉。要做嗎？
#

#!/usr/bin/env python3
from __future__ import annotations

"""
Train an EfficientNet-B0 heatmap regressor on ./cctag_dataset.

Required packages:
  pip install torch torchvision opencv-python numpy

Example:
  python train_cctag_heatmap.py --dataset_dir ./cctag_dataset --epochs 10 --batch_size 8
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet-B0 heatmap regressor for synthetic CCTag localization."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("./cctag_dataset"),
        help="Dataset root containing images/, heatmaps/, and labels.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./runs/cctag_heatmap"),
        help="Directory for checkpoints and training logs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="Train split ratio."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker count."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--input_width", type=int, default=640, help="Resized input width."
    )
    parser.add_argument(
        "--input_height", type=int, default=400, help="Resized input height."
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs in addition to best/last.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cpu, cuda, cuda:0. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for DataParallel, e.g. 0,1,2. Overrides --device when set.",
    )
    parser.add_argument(
        "--coord_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the soft-argmax coordinate regression loss term (0 to disable).",
    )
    return parser.parse_args()


def ensure_torchvision() -> tuple[Any, Any]:
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required for EfficientNet-B0.\n"
            "Install it with: pip install torchvision"
        ) from exc
    return efficientnet_b0, EfficientNet_B0_Weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class CCTagHeatmapDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset_dir: Path, input_size: tuple[int, int]) -> None:
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / "images"
        self.heatmaps_dir = dataset_dir / "heatmaps"
        self.labels_csv = dataset_dir / "labels.csv"
        self.input_width, self.input_height = input_size
        self.heatmap_width: int | None = None
        self.heatmap_height: int | None = None
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.heatmaps_dir.is_dir():
            raise FileNotFoundError(f"Missing heatmaps directory: {self.heatmaps_dir}")
        if not self.labels_csv.is_file():
            raise FileNotFoundError(f"Missing labels.csv: {self.labels_csv}")

        samples: list[dict[str, Any]] = []
        with self.labels_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row.get("filename", "").strip()
                if not filename:
                    continue

                image_path = self.images_dir / f"{filename}.png"
                heatmap_path = self.heatmaps_dir / f"{filename}.npy"
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image file: {image_path}")
                if not heatmap_path.is_file():
                    raise FileNotFoundError(f"Missing heatmap file: {heatmap_path}")
                heatmap_shape = np.load(heatmap_path, mmap_mode="r").shape
                if len(heatmap_shape) != 2:
                    raise ValueError(
                        f"Expected 2D heatmap, got shape {heatmap_shape} at {heatmap_path}"
                    )
                if self.heatmap_width is None or self.heatmap_height is None:
                    self.heatmap_height, self.heatmap_width = (
                        int(heatmap_shape[0]),
                        int(heatmap_shape[1]),
                    )
                elif heatmap_shape != (self.heatmap_height, self.heatmap_width):
                    raise ValueError(
                        f"Inconsistent heatmap shape: expected {(self.heatmap_height, self.heatmap_width)}, "
                        f"got {heatmap_shape} at {heatmap_path}"
                    )

                samples.append(
                    {
                        "filename": filename,
                        "image_path": image_path,
                        "heatmap_path": heatmap_path,
                        "center_x": float(row.get("center_x") or row.get("x") or 0.0),
                        "center_y": float(row.get("center_y") or row.get("y") or 0.0),
                        "occlusion_ratio": float(row.get("occlusion_ratio") or 0.0),
                    }
                )

        if not samples:
            raise ValueError(f"No samples found in {self.labels_csv}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        src_h, src_w = image.shape[:2]

        heatmap = np.load(sample["heatmap_path"]).astype(np.float32)
        if heatmap.ndim != 2:
            raise ValueError(
                f"Expected 2D heatmap, got shape {heatmap.shape} at {sample['heatmap_path']}"
            )

        image = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_AREA,
        )
        heatmap = np.clip(heatmap, 0.0, 1.0)

        image_scale_x = self.input_width / float(src_w)
        image_scale_y = self.input_height / float(src_h)
        center_x = float(sample["center_x"]) * image_scale_x
        center_y = float(sample["center_y"]) * image_scale_y
        heatmap_scale_x = float(self.heatmap_width) / float(src_w)
        heatmap_scale_y = float(self.heatmap_height) / float(src_h)
        center_x_heatmap = float(sample["center_x"]) * heatmap_scale_x
        center_y_heatmap = float(sample["center_y"]) * heatmap_scale_y

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        heatmap_tensor = torch.from_numpy(heatmap[None, ...]).float()
        center_tensor = torch.tensor([center_x, center_y], dtype=torch.float32)
        heatmap_center_tensor = torch.tensor(
            [center_x_heatmap, center_y_heatmap], dtype=torch.float32
        )

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "center": center_tensor,
            "heatmap_center": heatmap_center_tensor,
            "filename": sample["filename"],
            "occlusion_ratio": torch.tensor(
                sample["occlusion_ratio"], dtype=torch.float32
            ),
        }


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


class CCTagNet(nn.Module):
    def __init__(self, heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        efficientnet_b0, EfficientNet_B0_Weights = ensure_torchvision()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        prediction = self.decoder(features)
        return nn.functional.interpolate(
            prediction,
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=False,
        )


def split_dataset(
    dataset: Dataset[Any], train_ratio: float, seed: int
) -> tuple[Subset[Any], Subset[Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1, got {train_ratio}")

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_size = max(1, int(len(indices) * train_ratio))
    val_size = len(indices) - train_size
    if val_size == 0:
        train_size = len(indices) - 1
        val_size = 1

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader[Any], DataLoader[Any], int, tuple[int, int]]:
    dataset = CCTagHeatmapDataset(
        dataset_dir=args.dataset_dir,
        input_size=(args.input_width, args.input_height),
    )
    train_set, val_set = split_dataset(dataset, args.train_ratio, args.seed)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if dataset.heatmap_height is None or dataset.heatmap_width is None:
        raise ValueError("Failed to infer heatmap size from dataset")
    return (
        train_loader,
        val_loader,
        len(dataset),
        (dataset.heatmap_height, dataset.heatmap_width),
    )


def decode_heatmap_centers(heatmaps: torch.Tensor) -> torch.Tensor:
    """Sub-pixel center extraction via 2nd-order Taylor expansion around the argmax peak."""
    batch_size, _, height, width = heatmaps.shape
    hm = heatmaps.squeeze(1)  # [B, H, W]

    flat_indices = hm.view(batch_size, -1).argmax(dim=1)
    py = torch.div(flat_indices, width, rounding_mode="floor")
    px = flat_indices % width

    padded = nn.functional.pad(hm, (1, 1, 1, 1), mode="replicate")
    px_p = px + 1
    py_p = py + 1
    b = torch.arange(batch_size, device=hm.device)

    p11 = padded[b, py_p, px_p]
    p10 = padded[b, py_p, px_p - 1]
    p12 = padded[b, py_p, px_p + 1]
    p01 = padded[b, py_p - 1, px_p]
    p21 = padded[b, py_p + 1, px_p]

    dx = 0.5 * (p12 - p10)
    dy = 0.5 * (p21 - p01)
    dxx = p12 - 2.0 * p11 + p10
    dyy = p21 - 2.0 * p11 + p01
    dxy = 0.25 * (
        padded[b, py_p + 1, px_p + 1]
        - padded[b, py_p + 1, px_p - 1]
        - padded[b, py_p - 1, px_p + 1]
        + padded[b, py_p - 1, px_p - 1]
    )

    det = dxx * dyy - dxy * dxy
    valid = det.abs() > 1e-6

    delta_x = torch.zeros_like(dx)
    delta_y = torch.zeros_like(dy)
    delta_x[valid] = -(dyy[valid] * dx[valid] - dxy[valid] * dy[valid]) / det[valid]
    delta_y[valid] = -(-dxy[valid] * dx[valid] + dxx[valid] * dy[valid]) / det[valid]
    delta_x = delta_x.clamp(-0.5, 0.5)
    delta_y = delta_y.clamp(-0.5, 0.5)

    return torch.stack((px.float() + delta_x, py.float() + delta_y), dim=1)


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable (x, y) coordinate extraction via spatial softmax.

    Returns shape (B, 2) in heatmap pixel coordinates.
    Used during training to compute a differentiable coordinate loss.
    """
    batch_size, _, height, width = heatmaps.shape
    weights = torch.softmax(heatmaps.view(batch_size, -1), dim=1).view(
        batch_size, 1, height, width
    )
    x_grid = torch.arange(width, dtype=heatmaps.dtype, device=heatmaps.device).view(
        1, 1, 1, width
    )
    y_grid = torch.arange(height, dtype=heatmaps.dtype, device=heatmaps.device).view(
        1, 1, height, 1
    )
    x = (weights * x_grid).sum(dim=(1, 2, 3))
    y = (weights * y_grid).sum(dim=(1, 2, 3))
    return torch.stack((x, y), dim=1)


def compute_center_l2_px(
    prediction: torch.Tensor,
    target_centers_px: torch.Tensor,
    input_size: tuple[int, int],
) -> float:
    pred_centers = decode_heatmap_centers(prediction)
    input_width, input_height = input_size
    heatmap_height, heatmap_width = prediction.shape[-2:]
    scale_x = input_width / float(heatmap_width)
    scale_y = input_height / float(heatmap_height)
    pred_centers_px = pred_centers.clone()
    pred_centers_px[:, 0] *= scale_x
    pred_centers_px[:, 1] *= scale_y
    distances = torch.linalg.norm(pred_centers_px - target_centers_px, dim=1)
    return float(distances.mean().detach().cpu().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    coord_loss_weight: float = 1.0,
) -> tuple[float, float, float]:
    """Returns (total_loss, heatmap_loss, coord_loss) averaged over the epoch."""
    model.train()
    running_total = 0.0
    running_heatmap = 0.0
    running_coord = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        heatmap_centers = batch["heatmap_center"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)

        heatmap_loss = criterion(predictions, targets)

        hm_h, hm_w = predictions.shape[-2:]
        wh = torch.tensor([hm_w, hm_h], dtype=predictions.dtype, device=device)
        pred_coords_norm = soft_argmax_2d(predictions) / wh
        target_coords_norm = heatmap_centers / wh
        coord_loss = nn.functional.mse_loss(pred_coords_norm, target_coords_norm)

        loss = heatmap_loss + coord_loss_weight * coord_loss
        loss.backward()
        optimizer.step()

        n = images.size(0)
        running_total += tensor_to_float(loss) * n
        running_heatmap += tensor_to_float(heatmap_loss) * n
        running_coord += tensor_to_float(coord_loss) * n
        sample_count += n

    denom = max(sample_count, 1)
    return running_total / denom, running_heatmap / denom, running_coord / denom


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    input_size: tuple[int, int],
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_center_error = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        centers = batch["center"].to(device, non_blocking=True)

        predictions = model(images)
        loss = criterion(predictions, targets)
        center_error = compute_center_l2_px(predictions, centers, input_size)

        batch_size = images.size(0)
        running_loss += tensor_to_float(loss) * batch_size
        running_center_error += center_error * batch_size
        sample_count += batch_size

    avg_loss = running_loss / max(sample_count, 1)
    avg_center_error = running_center_error / max(sample_count, 1)
    return avg_loss, avg_center_error


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args),
        },
        checkpoint_path,
    )


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_size: int,
    train_size: int,
    val_size: int,
    device: torch.device,
    heatmap_size: tuple[int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_dir": str(args.dataset_dir),
        "dataset_size": dataset_size,
        "train_size": train_size,
        "val_size": val_size,
        "input_width": args.input_width,
        "input_height": args.input_height,
        "heatmap_width": heatmap_size[1],
        "heatmap_height": heatmap_size[0],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "device": str(device),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    train_heatmap_loss: float,
    train_coord_loss: float,
    val_loss: float,
    center_l2_px: float,
    lr: float,
    duration_sec: float,
) -> None:
    is_new = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_heatmap_loss",
                    "train_coord_loss",
                    "val_loss",
                    "center_l2_px",
                    "lr",
                    "duration_sec",
                ]
            )
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{train_heatmap_loss:.6f}",
                f"{train_coord_loss:.8f}",
                f"{val_loss:.6f}",
                f"{center_l2_px:.4f}",
                f"{lr:.8f}",
                f"{duration_sec:.2f}",
            ]
        )


def preview_batch(loader: DataLoader[Any]) -> None:
    batch = next(iter(loader))
    images = batch["image"]
    heatmaps = batch["heatmap"]
    centers = batch["center"]
    heatmap_centers = batch["heatmap_center"]
    print(
        "preview:",
        f"images={tuple(images.shape)}",
        f"heatmaps={tuple(heatmaps.shape)}",
        f"centers={tuple(centers.shape)}",
        f"heatmap_centers={tuple(heatmap_centers.shape)}",
        f"heatmap_range=({heatmaps.min().item():.4f}, {heatmaps.max().item():.4f})",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    gpu_ids: list[int] = []
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = resolve_device(args.device)

    train_loader, val_loader, dataset_size, heatmap_size = create_dataloaders(args)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    preview_batch(train_loader)

    model = CCTagNet(heatmap_size=heatmap_size).to(device)
    if gpu_ids and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"using DataParallel on GPUs: {gpu_ids}")
    criterion = CombinedHeatmapLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    write_run_config(
        output_dir=args.output_dir,
        args=args,
        dataset_size=dataset_size,
        train_size=train_size,
        val_size=val_size,
        device=device,
        heatmap_size=heatmap_size,
    )
    metrics_path = args.output_dir / "metrics.csv"

    best_val_loss = math.inf
    print(
        f"training on {device} | samples={dataset_size} train={train_size} val={val_size} "
        f"input={args.input_width}x{args.input_height} heatmap={heatmap_size[1]}x{heatmap_size[0]} "
        f"coord_loss_weight={args.coord_loss_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_heatmap_loss, train_coord_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.coord_loss_weight
        )
        val_loss, center_l2_px = validate(
            model,
            val_loader,
            criterion,
            device,
            input_size=(args.input_width, args.input_height),
        )
        scheduler.step(val_loss)
        duration_sec = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        append_metrics_row(
            metrics_path=metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            train_heatmap_loss=train_heatmap_loss,
            train_coord_loss=train_coord_loss,
            val_loss=val_loss,
            center_l2_px=center_l2_px,
            lr=current_lr,
            duration_sec=duration_sec,
        )
        save_checkpoint(
            checkpoint_path=args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_path=args.output_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_path=args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
            )

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train={train_loss:.6f} "
            f"(heatmap={train_heatmap_loss:.6f} coord={train_coord_loss:.6f}) "
            f"val={val_loss:.6f} "
            f"center_l2_px={center_l2_px:.3f} "
            f"lr={current_lr:.6g} "
            f"time={duration_sec:.1f}s"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("training interrupted", file=sys.stderr)
        raise SystemExit(130)
