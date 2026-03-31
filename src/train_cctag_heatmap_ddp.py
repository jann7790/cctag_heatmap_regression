#!/usr/bin/env python3
from __future__ import annotations

"""
Train an EfficientNet-B0 heatmap regressor with DistributedDataParallel.

Examples:
  torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
    --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
    --output_dir ./outputs/runs/experiment_mixed_ddp \
    --epochs 80 \
    --batch_size 18
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def spatial_soft_argmax(heatmap: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Differentiable soft-argmax returning (x, y) in heatmap pixel coords."""
    B, _, H, W = heatmap.shape
    flat = heatmap.view(B, -1) / temperature
    weights = torch.softmax(flat, dim=1).view(B, H, W)
    x_coords = torch.arange(W, dtype=torch.float32, device=heatmap.device)
    y_coords = torch.arange(H, dtype=torch.float32, device=heatmap.device)
    exp_x = (weights.sum(dim=1) * x_coords).sum(dim=1)
    exp_y = (weights.sum(dim=2) * y_coords).sum(dim=1)
    return torch.stack([exp_x, exp_y], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an EfficientNet-B0 heatmap regressor for synthetic CCTag localization with DDP."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("./outputs/datasets/cctag_dataset"),
        help="Dataset root containing images/, heatmaps/, and labels.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/runs/cctag_heatmap_ddp"),
        help="Directory for checkpoints and training logs.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-process mini-batch size. Global batch = batch_size * world_size.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count per process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--input_width", type=int, default=640, help="Resized input width.")
    parser.add_argument("--input_height", type=int, default=400, help="Resized input height.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save a checkpoint every N epochs in addition to best/last.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Distributed backend. Defaults to nccl on CUDA and gloo on CPU.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single-process fallback device, e.g. cpu, cuda, cuda:0.",
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


def resolve_single_process_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_ready() or dist.get_rank() == 0


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if is_dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


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
                    raise ValueError(f"Expected 2D heatmap, got shape {heatmap_shape} at {heatmap_path}")
                if self.heatmap_width is None or self.heatmap_height is None:
                    self.heatmap_height, self.heatmap_width = int(heatmap_shape[0]), int(heatmap_shape[1])
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
            raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape} at {sample['heatmap_path']}")

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
        heatmap_center_tensor = torch.tensor([center_x_heatmap, center_y_heatmap], dtype=torch.float32)

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "center": center_tensor,
            "heatmap_center": heatmap_center_tensor,
            "filename": sample["filename"],
            "occlusion_ratio": torch.tensor(sample["occlusion_ratio"], dtype=torch.float32),
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
    def __init__(self, center_weight: float = 0.1) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = HeatmapDiceLoss()
        self.center_weight = center_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        heatmap_centers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        heatmap_loss = 0.5 * self.bce(prediction, target) + 0.5 * self.dice(prediction, target)
        if heatmap_centers is not None:
            # Only apply center loss on positive samples (target heatmap has a real peak).
            # Negative samples should output a near-zero heatmap; forcing soft-argmax on
            # them causes the model to always predict a peak even for negatives.
            pos_mask = target.flatten(1).max(dim=1).values > 0.1  # (B,)
            if pos_mask.any():
                pred_centers = spatial_soft_argmax(prediction[pos_mask])
                center_loss = nn.functional.smooth_l1_loss(
                    pred_centers, heatmap_centers[pos_mask]
                )
                return heatmap_loss + self.center_weight * center_loss
        return heatmap_loss


class CCTagNet(nn.Module):
    """EfficientNet-B0 encoder with U-Net style skip connections.

    Skip taps from encoder stages (channel counts for B0):
        features[1] → 16ch, stride 2
        features[2] → 24ch, stride 4
        features[3] → 40ch, stride 8
        features[4] → 80ch, stride 16
    Bottleneck (features[8]) → 1280ch, stride 32
    """

    SKIP_INDICES = (1, 2, 3, 4)
    SKIP_CHANNELS = (16, 24, 40, 80)

    def __init__(self, heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        efficientnet_b0, EfficientNet_B0_Weights = ensure_torchvision()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size

        # Decoder: 4 upsample stages with skip-connection concatenation
        # stage 1: up 32→16, cat features[4] (80ch)
        self.dec1 = self._dec_block(1280 + 80, 256)
        # stage 2: up 16→8,  cat features[3] (40ch)
        self.dec2 = self._dec_block(256 + 40, 128)
        # stage 3: up 8→4,   cat features[2] (24ch)
        self.dec3 = self._dec_block(128 + 24, 64)
        # stage 4: up 4→2,   cat features[1] (16ch)
        self.dec4 = self._dec_block(64 + 16, 32)

        self.head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.SKIP_INDICES:
                skips.append(x)
        return x, skips  # x=1280ch; skips=[16ch, 24ch, 40ch, 80ch]

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, (s16, s24, s40, s80) = self._encode(x)

        x = self.dec1(self._up_cat(bottleneck, s80))
        x = self.dec2(self._up_cat(x, s40))
        x = self.dec3(self._up_cat(x, s24))
        x = self.dec4(self._up_cat(x, s16))
        x = self.head(x)

        return nn.functional.interpolate(
            x,
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=False,
        )


def split_dataset(dataset: Dataset[Any], train_ratio: float, seed: int) -> tuple[Subset[Any], Subset[Any]]:
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
    rank: int,
    world_size: int,
) -> tuple[DataLoader[Any], DataLoader[Any], int, tuple[int, int], DistributedSampler[Any] | None]:
    dataset = CCTagHeatmapDataset(
        dataset_dir=args.dataset_dir,
        input_size=(args.input_width, args.input_height),
    )
    train_set, val_set = split_dataset(dataset, args.train_ratio, args.seed)

    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    if dataset.heatmap_height is None or dataset.heatmap_width is None:
        raise ValueError("Failed to infer heatmap size from dataset")
    return train_loader, val_loader, len(dataset), (dataset.heatmap_height, dataset.heatmap_width), train_sampler


def decode_heatmap_centers(heatmaps: torch.Tensor) -> torch.Tensor:
    batch_size, _, _height, width = heatmaps.shape
    hm = heatmaps.squeeze(1)

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


def compute_center_distance_sum(
    prediction: torch.Tensor,
    target_centers_px: torch.Tensor,
    input_size: tuple[int, int],
) -> tuple[float, int]:
    pred_centers = decode_heatmap_centers(prediction)
    input_width, input_height = input_size
    heatmap_height, heatmap_width = prediction.shape[-2:]
    scale_x = input_width / float(heatmap_width)
    scale_y = input_height / float(heatmap_height)
    pred_centers_px = pred_centers.clone()
    pred_centers_px[:, 0] *= scale_x
    pred_centers_px[:, 1] *= scale_y
    distances = torch.linalg.norm(pred_centers_px - target_centers_px, dim=1)
    return float(distances.sum().detach().cpu().item()), int(distances.numel())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        heatmap_centers = batch["heatmap_center"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets, heatmap_centers=heatmap_centers)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += tensor_to_float(loss) * batch_size
        sample_count += batch_size

    total_loss = reduce_sum(running_loss, device)
    total_count = reduce_sum(float(sample_count), device)
    return total_loss / max(total_count, 1.0)


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
    running_center_distance = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        centers = batch["center"].to(device, non_blocking=True)

        predictions = model(images)
        loss = criterion(predictions, targets)
        center_distance_sum, distance_count = compute_center_distance_sum(predictions, centers, input_size)

        batch_size = images.size(0)
        running_loss += tensor_to_float(loss) * batch_size
        running_center_distance += center_distance_sum
        sample_count += distance_count

    total_loss = reduce_sum(running_loss, device)
    total_center_distance = reduce_sum(running_center_distance, device)
    total_count = reduce_sum(float(sample_count), device)
    avg_loss = total_loss / max(total_count, 1.0)
    avg_center_error = total_center_distance / max(total_count, 1.0)
    return avg_loss, avg_center_error


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    world_size: int,
) -> None:
    if not is_main_process():
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {**vars(args), "world_size": world_size},
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
    world_size: int,
) -> None:
    if not is_main_process():
        return
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
        "batch_size_per_process": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "device": str(device),
        "world_size": world_size,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    center_l2_px: float,
    lr: float,
    duration_sec: float,
) -> None:
    if not is_main_process():
        return
    is_new = not metrics_path.exists()
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "center_l2_px", "lr", "duration_sec"]
            )
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{center_l2_px:.4f}",
                f"{lr:.8f}",
                f"{duration_sec:.2f}",
            ]
        )


def preview_batch(loader: DataLoader[Any]) -> None:
    if not is_main_process():
        return
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


def setup_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but CUDA is not available.")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        return rank, world_size, local_rank, device

    device = resolve_single_process_device(args.device)
    return 0, 1, 0, device


def cleanup_distributed() -> None:
    if is_dist_ready():
        dist.destroy_process_group()


def build_model(heatmap_size: tuple[int, int], device: torch.device, rank: int, world_size: int) -> nn.Module:
    # Rank 0 loads pretrained weights first so other ranks can reuse the local cache.
    if world_size > 1 and rank != 0:
        dist.barrier()
    model = CCTagNet(heatmap_size=heatmap_size).to(device)
    if world_size > 1 and rank == 0:
        dist.barrier()
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    return model


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed(args)
    set_seed(args.seed + rank)

    train_loader, val_loader, dataset_size, heatmap_size, train_sampler = create_dataloaders(
        args,
        rank=rank,
        world_size=world_size,
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    preview_batch(train_loader)

    model = build_model(heatmap_size=heatmap_size, device=device, rank=rank, world_size=world_size)
    criterion = CombinedHeatmapLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    write_run_config(
        output_dir=args.output_dir,
        args=args,
        dataset_size=dataset_size,
        train_size=train_size,
        val_size=val_size,
        device=device,
        heatmap_size=heatmap_size,
        world_size=world_size,
    )
    metrics_path = args.output_dir / "metrics.csv"

    best_val_loss = math.inf
    if is_main_process():
        print(
            f"training on {device} | world_size={world_size} | "
            f"samples={dataset_size} train={train_size} val={val_size} "
            f"input={args.input_width}x{args.input_height} heatmap={heatmap_size[1]}x{heatmap_size[0]} "
            f"per_gpu_batch={args.batch_size} global_batch={args.batch_size * world_size}"
        )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, center_l2_px = validate(
            model,
            val_loader,
            criterion,
            device,
            input_size=(args.input_width, args.input_height),
        )
        scheduler.step()
        duration_sec = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        append_metrics_row(
            metrics_path=metrics_path,
            epoch=epoch,
            train_loss=train_loss,
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
            world_size=world_size,
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
                world_size=world_size,
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
                world_size=world_size,
            )

        if is_main_process():
            print(
                f"epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"center_l2_px={center_l2_px:.3f} "
                f"lr={current_lr:.6g} "
                f"time={duration_sec:.1f}s"
            )

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if is_main_process():
            print("training interrupted", file=sys.stderr)
        cleanup_distributed()
        raise SystemExit(130)
