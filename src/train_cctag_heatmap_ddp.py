#!/usr/bin/env python3
from __future__ import annotations

"""
Train a backbone-selectable heatmap regressor with DistributedDataParallel.

Examples:
  torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
    --dataset_dir ./outputs/training_sets/generated_training_sets/mixed_train_dataset \
    --output_dir ./outputs/runs/experiment_mixed_ddp \
    --epochs 80 \
    --batch_size 18

  torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
    --train_dataset_dir ./outputs/training_sets/stride4_v2/mixed_train_dataset \
    --train_dataset_dir ./outputs/real_world_stride4_train \
    --val_dataset_dir ./outputs/real_world_stride4_test \
    --output_dir ./outputs/runs/experiment_multi_source_ddp \
    --epochs 80 \
    --batch_size 18
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

# Floor for ellipse semi-axes before taking log() in the size target.
# Real positive samples have ellipse_a/b >> 1 px; this only matters for
# negative rows where the value is 0 (those are masked out by the loss anyway).
SIZE_TARGET_EPS = 1e-3

# Ablation toggle for the decoder's bilinear up/down-sampling. PyTorch's default
# (align_corners=False) shifts the sampling grid by a fraction of a pixel at each
# interpolate stage; the cumulative effect is a candidate cause of the systematic
# vertical center-decode bias. main() overrides this from --align_corners.
DECODE_ALIGN_CORNERS = False


def resolve_heatmap_path(directory: Path, stem: str) -> Path:
    """Heatmap file for a stem, preferring compressed .npz over legacy .npy."""
    npz = directory / f"{stem}.npz"
    return npz if npz.exists() else directory / f"{stem}.npy"


def load_heatmap(path: Path) -> np.ndarray:
    """Load a heatmap stored as compressed .npz (key 'heatmap') or legacy .npy, as float32."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data["heatmap"].astype(np.float32)
    return np.load(path).astype(np.float32)


@dataclass
class AugmentConfig:
    """Online train-time augmentation. Photometric ops touch the image only;
    horizontal flip also mirrors the heatmap and the center x-coordinates so the
    label stays consistent. Ellipse a/b and the size target are flip-invariant."""

    hflip_prob: float = 0.5
    brightness: float = 0.2  # +/- fraction of 255
    contrast: float = 0.2  # factor in [1-c, 1+c]
    noise_std: float = 6.0  # max gaussian sigma in 0-255 space


def spatial_soft_argmax(
    heatmap: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
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
        description="Train a heatmap regressor for synthetic CCTag localization with DDP."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("./outputs/datasets/cctag_dataset"),
        help="Single dataset root. Used as a fallback when explicit train/val dataset dirs are not provided.",
    )
    parser.add_argument(
        "--train_dataset_dir",
        dest="train_dataset_dirs",
        action="append",
        type=Path,
        default=[],
        help="Training dataset root. Repeat this flag to train from multiple dataset sources.",
    )
    parser.add_argument(
        "--val_dataset_dir",
        dest="val_dataset_dirs",
        action="append",
        type=Path,
        default=[],
        help="Validation dataset root. Repeat this flag to validate on multiple dataset sources. "
        "Optional: if omitted while --train_dataset_dir is given, a val split is carved out of "
        "the merged training data using --train_ratio.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/runs/cctag_heatmap_ddp"),
        help="Directory for checkpoints and training logs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-process mini-batch size. Global batch = batch_size * world_size.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio. Used for the internal val split when no explicit "
        "--val_dataset_dir is given (single --dataset_dir, or --train_dataset_dir without --val_dataset_dir).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker count per process.",
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
        default=10,
        help="Save a numbered epoch_NNN.pt checkpoint every N epochs (best.pt and "
        "last.pt are always updated every epoch regardless).",
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
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=[
            "efficientnet_b0",
            "mobilenet_v3_small",
            "resnet18",
            "resnet18_hires",
            "hrnet_w18",
        ],
        help="Backbone architecture (default: resnet18). 'resnet18_hires' drops "
        "the stem maxpool (2x finer encoder) and uses a double-conv decoder for "
        "sub-pixel localization; its checkpoints are not interchangeable with "
        "'resnet18'. 'hrnet_w18' uses timm ImageNet-pretrained HRNetV2 features "
        "fused at stride 4.",
    )
    parser.add_argument(
        "--align_corners",
        action="store_true",
        help="Use align_corners=True in the decoder's bilinear interpolate stages "
        "(ablation for the systematic vertical center-decode bias). Default: False "
        "(PyTorch default). Changes spatial alignment, so checkpoints are not "
        "interchangeable across this flag.",
    )
    parser.add_argument(
        "--focal_loss",
        action="store_true",
        help="Replace BCE with Focal Loss to focus on hard false-positive pixels.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha (positive class weight). Default: 0.25.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (focusing parameter). Default: 2.0.",
    )
    parser.add_argument(
        "--ohem_ratio",
        type=float,
        default=0.0,
        help="Online Hard Example Mining: keep only the hardest K%% of pixels for loss. "
        "0.0 disables OHEM; 0.3 keeps the hardest 30%%. Only used when --focal_loss is not set.",
    )
    parser.add_argument(
        "--offset_head",
        action="store_true",
        help="Add a 2-channel offset regression head (CenterNet-style) for sub-pixel center decoding.",
    )
    parser.add_argument(
        "--offset_weight",
        type=float,
        default=1.0,
        help="Weight of the offset L1 loss term. Default: 1.0.",
    )
    parser.add_argument(
        "--offset_head_hidden",
        type=int,
        default=0,
        help="Hidden channels for the offset head. 0 (default) = a single 1x1 conv "
        "(legacy). >0 inserts a 3x3 conv + ReLU before the 1x1 (e.g. 32/64) for "
        "higher sub-pixel capacity, directly targeting center-L2.",
    )
    parser.add_argument(
        "--occ_loss_weight",
        type=float,
        default=0.0,
        help="Per-sample heatmap-loss weighting by occlusion: weight = 1 + k*occ_ratio "
        "(mean-normalized, so overall loss scale is unchanged). Focuses learning on "
        "hard-occluded markers. Default: 0.0 (off). Try 1.0 for gentle weighting.",
    )
    parser.add_argument(
        "--size_head",
        action="store_true",
        help="Add a 2-channel size head that regresses log(ellipse_a), log(ellipse_b) "
        "at the predicted center (source-image pixels).",
    )
    parser.add_argument(
        "--size_weight",
        type=float,
        default=1.0,
        help="Weight of the size smooth-L1 loss term. Default: 1.0.",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Path to a checkpoint (.pt). Loads model weights into the new run; "
        "optimizer/scheduler/epoch counter start fresh.",
    )
    # --- performance (Tier 2): all default OFF/identity, no behavior change unless set ---
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (fp16 autocast on the forward pass + "
        "GradScaler) on CUDA. ~1.5-2x faster and ~half the activation memory. The loss "
        "is still computed in fp32 (BCELoss is unsafe under autocast). Only affects "
        "the training forward/backward; validation always runs fp32 so center_l2_px is "
        "not corrupted by the fp16 peak-plateau argmax bias.",
    )
    parser.add_argument(
        "--channels_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use channels_last memory format for the model and inputs (faster conv on "
        "Tensor Cores). Numerically equivalent. ON by default; disable with "
        "--no-channels_last.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Wrap the model in torch.compile (PyTorch 2.x graph fusion). One-time warmup "
        "cost; best with fixed input size (which this training uses).",
    )
    parser.add_argument(
        "--no_cudnn_benchmark",
        action="store_true",
        help="Disable cudnn.benchmark autotuning. It is ON by default here because the "
        "input size is fixed; turn off only for strict determinism.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the TF32 Tensor Core path for conv/matmul on Ampere+ GPUs "
        "(fp32 accumulate; effectively lossless for CNN training). ON by default; "
        "disable with --no-tf32.",
    )
    # --- online augmentation (Tier 3): OFF unless --augment ---
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable online train-time augmentation (photometric + horizontal flip). "
        "Validation stays deterministic.",
    )
    parser.add_argument(
        "--aug_hflip_prob",
        type=float,
        default=0.5,
        help="Horizontal-flip probability when --augment. Default: 0.5.",
    )
    parser.add_argument(
        "--aug_brightness",
        type=float,
        default=0.2,
        help="Max brightness jitter as a fraction of 255 (uniform +/-) when --augment.",
    )
    parser.add_argument(
        "--aug_contrast",
        type=float,
        default=0.2,
        help="Max contrast jitter (factor 1 +/- this) when --augment.",
    )
    parser.add_argument(
        "--aug_noise_std",
        type=float,
        default=6.0,
        help="Max gaussian noise std (0-255 space) when --augment.",
    )
    args = parser.parse_args()

    # HRNet requires input dimensions divisible by 32 (5 × stride-2 stages).
    # Auto-adjust with a warning so the user doesn't hit a cryptic size-mismatch
    # RuntimeError inside timm's hrnet.py stage forward.
    if "hrnet" in args.backbone.lower():
        def _round_up_32(val: int) -> int:
            return ((val + 31) // 32) * 32

        w_ok = _round_up_32(args.input_width)
        h_ok = _round_up_32(args.input_height)
        if args.input_width != w_ok or args.input_height != h_ok:
            print(
                f"[HRNet] input dimensions must be multiples of 32 "
                f"(got {args.input_width}x{args.input_height}). "
                f"Auto-adjusting to {w_ok}x{h_ok}."
            )
            args.input_width = w_ok
            args.input_height = h_ok

    return args


def ensure_torchvision() -> tuple[Any, Any]:
    try:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required for EfficientNet-B0.\n"
            "Install it with: pip install torchvision"
        ) from exc
    return efficientnet_b0, EfficientNet_B0_Weights


def ensure_mobilenet_v3_small() -> tuple[Any, Any]:
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required for MobileNetV3-Small.\n"
            "Install it with: pip install torchvision"
        ) from exc
    return mobilenet_v3_small, MobileNet_V3_Small_Weights


def ensure_resnet18() -> tuple[Any, Any]:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required for ResNet-18.\n"
            "Install it with: pip install torchvision"
        ) from exc
    return resnet18, ResNet18_Weights


def ensure_timm() -> Any:
    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            "timm is required for HRNet backbones.\n"
            "Install it with: uv sync (timm is now a dependency) or pip install timm"
        ) from exc
    return timm


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
    def __init__(
        self,
        dataset_dir: Path,
        input_size: tuple[int, int],
        augment: AugmentConfig | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / "images"
        self.heatmaps_dir = dataset_dir / "heatmaps"
        self.labels_csv = dataset_dir / "labels.csv"
        self.input_width, self.input_height = input_size
        self.augment = augment
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
                heatmap_path = resolve_heatmap_path(self.heatmaps_dir, filename)
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image file: {image_path}")
                if not heatmap_path.is_file():
                    raise FileNotFoundError(f"Missing heatmap file: {heatmap_path}")
                heatmap_shape = load_heatmap(heatmap_path).shape
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
                        "ellipse_a": float(row.get("ellipse_a") or 0.0),
                        "ellipse_b": float(row.get("ellipse_b") or 0.0),
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

        heatmap = load_heatmap(sample["heatmap_path"])
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

        if self.augment is not None:
            (
                image,
                heatmap,
                center_x,
                center_y,
                center_x_heatmap,
                center_y_heatmap,
            ) = self._apply_augment(
                image, heatmap, center_x, center_y, center_x_heatmap, center_y_heatmap
            )

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        heatmap_tensor = torch.from_numpy(heatmap[None, ...]).float()
        center_tensor = torch.tensor([center_x, center_y], dtype=torch.float32)
        heatmap_center_tensor = torch.tensor(
            [center_x_heatmap, center_y_heatmap], dtype=torch.float32
        )

        # Size target in log-space (source-image pixels). EPS floor guards
        # against log(0) on negative/degenerate rows; the loss masks them out
        # via the positive-sample mask anyway.
        ellipse_a = max(float(sample["ellipse_a"]), SIZE_TARGET_EPS)
        ellipse_b = max(float(sample["ellipse_b"]), SIZE_TARGET_EPS)
        size_target_tensor = torch.tensor(
            [math.log(ellipse_a), math.log(ellipse_b)], dtype=torch.float32
        )

        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "center": center_tensor,
            "heatmap_center": heatmap_center_tensor,
            "size_target": size_target_tensor,
            "filename": sample["filename"],
            "occlusion_ratio": torch.tensor(
                sample["occlusion_ratio"], dtype=torch.float32
            ),
        }

    def _apply_augment(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        center_x: float,
        center_y: float,
        cx_hm: float,
        cy_hm: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Photometric jitter (image only) + horizontal flip (image + heatmap +
        x-coords). Returns contiguous float32 arrays safe for torch.from_numpy."""
        cfg = self.augment
        assert cfg is not None
        img = image.astype(np.float32)

        if random.random() < cfg.hflip_prob:
            img = img[:, ::-1, :]
            heatmap = heatmap[:, ::-1]
            center_x = (self.input_width - 1) - center_x
            cx_hm = (self.heatmap_width - 1) - cx_hm

        if cfg.contrast > 0:
            factor = 1.0 + random.uniform(-cfg.contrast, cfg.contrast)
            mean = float(img.mean())
            img = (img - mean) * factor + mean
        if cfg.brightness > 0:
            img = img + random.uniform(-cfg.brightness, cfg.brightness) * 255.0
        if cfg.noise_std > 0:
            sigma = random.uniform(0.0, cfg.noise_std)
            if sigma > 0:
                img = img + np.random.normal(0.0, sigma, img.shape).astype(np.float32)

        img = np.ascontiguousarray(np.clip(img, 0.0, 255.0), dtype=np.float32)
        heatmap = np.ascontiguousarray(heatmap, dtype=np.float32)
        return img, heatmap, center_x, center_y, cx_hm, cy_hm


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


class HeatmapFocalLoss(nn.Module):
    """Focal loss for heatmap regression.

    Down-weights easy background pixels and focuses learning on hard
    false-positive-prone pixels (e.g. edges, curves, overexposed regions).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        pred = prediction.clamp(eps, 1.0 - eps)
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        pt = target * pred + (1.0 - target) * (1.0 - pred)
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_weight = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


class CombinedHeatmapLoss(nn.Module):
    def __init__(
        self,
        center_weight: float = 0.1,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ohem_ratio: float = 0.0,
        offset_weight: float = 1.0,
        size_weight: float = 1.0,
        occ_weight_k: float = 0.0,
    ) -> None:
        super().__init__()
        if use_focal:
            self.pixel_loss = HeatmapFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.pixel_loss = nn.BCELoss(reduction="none" if ohem_ratio > 0 else "mean")
        self.dice = HeatmapDiceLoss()
        self.center_weight = center_weight
        self.use_focal = use_focal
        self.ohem_ratio = ohem_ratio
        self.offset_weight = offset_weight
        self.size_weight = size_weight
        # Per-sample heatmap-loss weighting by occlusion: w = 1 + k * occ_ratio,
        # mean-normalized so the overall loss scale (hence effective LR) is
        # unchanged. k=0 disables it. Focuses learning on hard-occluded markers.
        self.occ_weight_k = occ_weight_k

    def _pixel_term(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.ohem_ratio > 0 and not self.use_focal:
            per_pixel = self.pixel_loss(prediction, target)
            flat = per_pixel.flatten()
            k = max(1, int(flat.numel() * self.ohem_ratio))
            topk_loss, _ = torch.topk(flat, k)
            return topk_loss.mean()
        return self.pixel_loss(prediction, target)

    def _weighted_heatmap_term(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        occlusion_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample 0.5*pixel + 0.5*dice, weighted by (1 + k*occ_ratio).

        Weights are mean-normalized so the batch loss scale (and effective LR)
        match the unweighted path; only the relative emphasis changes.
        """
        eps = 1e-7
        pred = prediction.clamp(eps, 1.0 - eps)
        if self.use_focal:
            bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
            pt = target * pred + (1.0 - target) * (1.0 - pred)
            focal_weight = (1.0 - pt) ** self.pixel_loss.gamma
            alpha_weight = (
                target * self.pixel_loss.alpha
                + (1.0 - target) * (1.0 - self.pixel_loss.alpha)
            )
            px = (alpha_weight * focal_weight * bce).flatten(1).mean(dim=1)  # (B,)
        else:
            px = nn.functional.binary_cross_entropy(
                pred, target, reduction="none"
            ).flatten(1).mean(dim=1)
        p2, t2 = prediction.flatten(1), target.flatten(1)
        inter = (p2 * t2).sum(dim=1)
        den = p2.sum(dim=1) + t2.sum(dim=1)
        dice = 1.0 - (2.0 * inter + self.dice.smooth) / (den + self.dice.smooth)  # (B,)
        per_sample = 0.5 * px + 0.5 * dice
        w = 1.0 + self.occ_weight_k * occlusion_ratio
        w = w / w.mean().clamp(min=eps)
        return (w * per_sample).mean()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        heatmap_centers: torch.Tensor | None = None,
        offset_pred: torch.Tensor | None = None,
        size_pred: torch.Tensor | None = None,
        size_target: torch.Tensor | None = None,
        occlusion_ratio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.occ_weight_k > 0.0 and occlusion_ratio is not None:
            heatmap_loss = self._weighted_heatmap_term(
                prediction, target, occlusion_ratio
            )
        else:
            heatmap_loss = 0.5 * self._pixel_term(prediction, target) + 0.5 * self.dice(
                prediction, target
            )
        total = heatmap_loss

        # Sub-pixel offset L1 loss (CenterNet-style), positives only. Supervise the
        # offset over a 3x3 neighborhood around the GT peak (not just round(center)):
        # at inference the decoder reads offset at the argmax peak, which need not
        # equal round(center), so the single-pixel target left those cells
        # unsupervised. Each neighbor cell predicts (heatmap_center - its own int
        # coord); out-of-bounds neighbors are masked out.
        if offset_pred is not None and heatmap_centers is not None:
            pos_mask = target.flatten(1).max(dim=1).values > 0.1
            if pos_mask.any():
                centers_pos = heatmap_centers[pos_mask]  # (P, 2) [x, y]
                offset_pos = offset_pred[pos_mask]  # (P, 2, H, W)
                P, _, H, W = offset_pos.shape
                peak_int = centers_pos.round().long()  # (P, 2) [x, y]
                p_idx = torch.arange(P, device=offset_pos.device)
                preds: list[torch.Tensor] = []
                tgts: list[torch.Tensor] = []
                for ddy in (-1, 0, 1):
                    for ddx in (-1, 0, 1):
                        nx = peak_int[:, 0] + ddx
                        ny = peak_int[:, 1] + ddy
                        valid = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
                        if not valid.any():
                            continue
                        nxv = nx[valid]
                        nyv = ny[valid]
                        preds.append(offset_pos[p_idx[valid], :, nyv, nxv])  # (V, 2)
                        tgts.append(
                            centers_pos[valid]
                            - torch.stack((nxv, nyv), dim=1).float()
                        )
                offset_loss = nn.functional.smooth_l1_loss(
                    torch.cat(preds), torch.cat(tgts)
                )
                total = total + self.offset_weight * offset_loss

        # Scale (ellipse a/b in log-space) smooth-L1 loss at the GT peak, positives only.
        if (
            size_pred is not None
            and size_target is not None
            and heatmap_centers is not None
        ):
            pos_mask = target.flatten(1).max(dim=1).values > 0.1
            if pos_mask.any():
                centers_pos = heatmap_centers[pos_mask]
                size_pos = size_pred[pos_mask]  # (P, 2, H, W)
                target_pos = size_target[pos_mask]  # (P, 2) in log-space
                _, _, H, W = size_pos.shape
                peak_int = centers_pos.round().long()
                peak_int[:, 0].clamp_(0, W - 1)
                peak_int[:, 1].clamp_(0, H - 1)
                p_idx = torch.arange(size_pos.shape[0], device=size_pos.device)
                size_at_peak = size_pos[
                    p_idx, :, peak_int[:, 1], peak_int[:, 0]
                ]  # (P, 2)
                size_loss = nn.functional.smooth_l1_loss(size_at_peak, target_pos)
                total = total + self.size_weight * size_loss

        if heatmap_centers is not None:
            # Soft-argmax center loss (legacy term, only on positive samples).
            pos_mask = target.flatten(1).max(dim=1).values > 0.1
            if pos_mask.any():
                pred_centers = spatial_soft_argmax(prediction[pos_mask])
                center_loss = nn.functional.smooth_l1_loss(
                    pred_centers, heatmap_centers[pos_mask]
                )
                total = total + self.center_weight * center_loss
        return total


def make_offset_head(in_ch: int, hidden: int = 0) -> nn.Module:
    """Sub-pixel offset (dx, dy) regression head.

    hidden <= 0 -> a single 1x1 conv (legacy). Keeps state_dict keys
                   ``offset_head.weight`` / ``offset_head.bias`` so checkpoints
                   trained before this option still load unchanged.
    hidden  > 0 -> 3x3(in_ch->hidden) + ReLU + 1x1(hidden->2). A higher-capacity
                   head: the only mechanism that recovers sub-stride precision, so
                   widening it directly targets center-L2 without touching the
                   backbone (keys become ``offset_head.0.*`` / ``offset_head.2.*``).
    """
    if hidden <= 0:
        return nn.Conv2d(in_ch, 2, kernel_size=1)
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, 2, kernel_size=1),
    )


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

    def __init__(
        self,
        heatmap_size: tuple[int, int],
        use_offset_head: bool = False,
        use_size_head: bool = False,
        offset_head_hidden: int = 0,
    ) -> None:
        super().__init__()
        efficientnet_b0, EfficientNet_B0_Weights = ensure_torchvision()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.use_offset_head = use_offset_head
        self.use_size_head = use_size_head

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
        if use_offset_head:
            self.offset_head = make_offset_head(32, offset_head_hidden)
        if use_size_head:
            self.size_head = nn.Conv2d(32, 2, kernel_size=1)

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
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
        )
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
            )
        return torch.cat([x, skip], dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bottleneck, (s16, s24, s40, s80) = self._encode(x)

        x = self.dec1(self._up_cat(bottleneck, s80))
        x = self.dec2(self._up_cat(x, s40))
        x = self.dec3(self._up_cat(x, s24))
        x = self.dec4(self._up_cat(x, s16))
        feat = x

        heatmap = nn.functional.interpolate(
            self.head(feat),
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=DECODE_ALIGN_CORNERS,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


class CCTagNetMobileV3(nn.Module):
    """MobileNetV3-Small encoder with U-Net style skip connections.

    Skip taps from encoder stages (channel counts for MobileNetV3-Small):
        features[0] → 16ch, stride 2
        features[1] → 16ch, stride 4
        features[3] → 24ch, stride 8
        features[8] → 48ch, stride 16
    Bottleneck (features[12]) → 576ch, stride 32
    """

    SKIP_INDICES = (0, 1, 3, 8)
    SKIP_CHANNELS = (16, 16, 24, 48)
    BOTTLENECK_CHANNELS = 576

    def __init__(
        self,
        heatmap_size: tuple[int, int],
        use_offset_head: bool = False,
        use_size_head: bool = False,
        offset_head_hidden: int = 0,
    ) -> None:
        super().__init__()
        mobilenet_v3_small, MobileNet_V3_Small_Weights = ensure_mobilenet_v3_small()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.use_offset_head = use_offset_head
        self.use_size_head = use_size_head

        self.dec1 = self._dec_block(self.BOTTLENECK_CHANNELS + 48, 128)
        self.dec2 = self._dec_block(128 + 24, 64)
        self.dec3 = self._dec_block(64 + 16, 32)
        self.dec4 = self._dec_block(32 + 16, 16)
        self.head = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid())
        if use_offset_head:
            self.offset_head = make_offset_head(16, offset_head_hidden)
        if use_size_head:
            self.size_head = nn.Conv2d(16, 2, kernel_size=1)

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
        return x, skips  # x=576ch; skips=[16ch, 16ch, 24ch, 48ch]

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
        )
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
            )
        return torch.cat([x, skip], dim=1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bottleneck, (s16_s2, s16_s4, s24, s48) = self._encode(x)
        x = self.dec1(self._up_cat(bottleneck, s48))
        x = self.dec2(self._up_cat(x, s24))
        x = self.dec3(self._up_cat(x, s16_s4))
        x = self.dec4(self._up_cat(x, s16_s2))
        feat = x
        heatmap = nn.functional.interpolate(
            self.head(feat),
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=DECODE_ALIGN_CORNERS,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


class CCTagNetResNet18(nn.Module):
    """ResNet-18 encoder with U-Net style skip connections.

    Skip taps:
        relu(conv1) → 64ch, stride 2
        layer1 → 64ch, stride 4
        layer2 → 128ch, stride 8
        layer3 → 256ch, stride 16
    Bottleneck (layer4) → 512ch, stride 32
    """

    def __init__(
        self,
        heatmap_size: tuple[int, int],
        use_offset_head: bool = False,
        use_size_head: bool = False,
        offset_head_hidden: int = 0,
    ) -> None:
        super().__init__()
        resnet18, ResNet18_Weights = ensure_resnet18()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.use_offset_head = use_offset_head
        self.use_size_head = use_size_head

        self.dec1 = self._dec_block(512 + 256, 256)
        self.dec2 = self._dec_block(256 + 128, 128)
        self.dec3 = self._dec_block(128 + 64, 64)
        self.dec4 = self._dec_block(64 + 64, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1), nn.Sigmoid())
        if use_offset_head:
            self.offset_head = make_offset_head(32, offset_head_hidden)
        if use_size_head:
            self.size_head = nn.Conv2d(32, 2, kernel_size=1)

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
        )
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
            )
        return torch.cat([x, skip], dim=1)

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        s64_s2 = self.stem(x)
        x = self.maxpool(s64_s2)
        s64_s4 = self.layer1(x)
        s128_s8 = self.layer2(s64_s4)
        s256_s16 = self.layer3(s128_s8)
        bottleneck = self.layer4(s256_s16)
        return bottleneck, (s64_s2, s64_s4, s128_s8, s256_s16)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bottleneck, (s64_s2, s64_s4, s128_s8, s256_s16) = self._encode(x)
        x = self.dec1(self._up_cat(bottleneck, s256_s16))
        x = self.dec2(self._up_cat(x, s128_s8))
        x = self.dec3(self._up_cat(x, s64_s4))
        x = self.dec4(self._up_cat(x, s64_s2))
        feat = x
        heatmap = nn.functional.interpolate(
            self.head(feat),
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=DECODE_ALIGN_CORNERS,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


class CCTagNetResNet18HiRes(nn.Module):
    """ResNet-18 encoder (stem maxpool dropped) with a higher-capacity decoder.

    Variant of CCTagNetResNet18 aimed at sub-pixel center localization. Dropping
    the stem maxpool keeps the encoder pyramid 2x finer (bottleneck stride 16
    instead of 32), so the precise peak location is not discarded by the early
    max-pool; the decoder uses double-conv blocks (two 3x3 convs per stage) so it
    has the capacity to exploit the finer skips. The final decoder feature stays
    at stride 2 and the output heatmap resolution is unchanged (set by
    heatmap_size) -- this deliberately isolates "finer encoder + bigger decoder"
    from the already-tried (and worse) finer-output-heatmap experiment.

    Skip taps (conv1 keeps stride 2, maxpool removed):
        layer1 -> 64ch,  stride 2
        layer2 -> 128ch, stride 4
        layer3 -> 256ch, stride 8
    Bottleneck (layer4) -> 512ch, stride 16
    """

    def __init__(
        self,
        heatmap_size: tuple[int, int],
        use_offset_head: bool = False,
        use_size_head: bool = False,
        offset_head_hidden: int = 0,
    ) -> None:
        super().__init__()
        resnet18, ResNet18_Weights = ensure_resnet18()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        # backbone.maxpool is intentionally dropped to keep early resolution.
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.use_offset_head = use_offset_head
        self.use_size_head = use_size_head

        self.dec1 = self._dec_block(512 + 256, 256)
        self.dec2 = self._dec_block(256 + 128, 128)
        self.dec3 = self._dec_block(128 + 64, 64)
        self.head = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.Sigmoid())
        if use_offset_head:
            self.offset_head = make_offset_head(64, offset_head_hidden)
        if use_size_head:
            self.size_head = nn.Conv2d(64, 2, kernel_size=1)

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        # Double conv: more decoder capacity than the single-conv block in
        # CCTagNetResNet18, so the finer encoder skips can actually be used.
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
        )
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(
                skip, size=x.shape[2:], mode="bilinear", align_corners=DECODE_ALIGN_CORNERS
            )
        return torch.cat([x, skip], dim=1)

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        s_stem = self.stem(x)  # stride 2, 64ch
        s64_s2 = self.layer1(s_stem)  # stride 2, 64ch (no maxpool)
        s128_s4 = self.layer2(s64_s2)  # stride 4, 128ch
        s256_s8 = self.layer3(s128_s4)  # stride 8, 256ch
        bottleneck = self.layer4(s256_s8)  # stride 16, 512ch
        return bottleneck, (s64_s2, s128_s4, s256_s8)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bottleneck, (s64_s2, s128_s4, s256_s8) = self._encode(x)
        x = self.dec1(self._up_cat(bottleneck, s256_s8))
        x = self.dec2(self._up_cat(x, s128_s4))
        x = self.dec3(self._up_cat(x, s64_s2))
        feat = x
        heatmap = nn.functional.interpolate(
            self.head(feat),
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=DECODE_ALIGN_CORNERS,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


class CCTagNetHRNet(nn.Module):
    """HRNet backbone with a stride-4 HRNetV2 fusion head for heatmap regression.

    Uses timm's features_only API to get HRNet's multi-resolution branch outputs,
    upsamples them all to the finest branch, concatenates them, and applies a
    compact convolutional fusion head. Channel dims come from feature_info so
    the class is not tied to one hardcoded HRNet channel layout.
    """

    def __init__(
        self,
        heatmap_size: tuple[int, int],
        variant: str = "hrnet_w18",
        use_offset_head: bool = False,
        use_size_head: bool = False,
        pretrained: bool = True,
        offset_head_hidden: int = 0,
    ) -> None:
        super().__init__()
        timm = ensure_timm()
        # out_indices=(1,2,3,4) drops timm's stride-2 stem feature; we fuse the
        # four HRNet branches at stride 4 (the documented HRNetV2 head). Keeping
        # the stem would fuse 1984 ch at half-res (320x512), ~4x the activation
        # memory and a different resolution than the heatmap target.
        self.encoder = timm.create_model(
            variant, pretrained=pretrained, features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        sum_ch = sum(self.encoder.feature_info.channels())
        fuse_ch = 128
        self.fuse = nn.Sequential(
            nn.Conv2d(sum_ch, fuse_ch, kernel_size=1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, fuse_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(fuse_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Conv2d(fuse_ch, 1, kernel_size=1), nn.Sigmoid())
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.use_offset_head = use_offset_head
        self.use_size_head = use_size_head
        if use_offset_head:
            self.offset_head = make_offset_head(fuse_ch, offset_head_hidden)
        if use_size_head:
            self.size_head = nn.Conv2d(fuse_ch, 2, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        feats = self.encoder(x)
        target = feats[0].shape[2:]
        ups = [feats[0]]
        ups.extend(
            nn.functional.interpolate(
                feat,
                size=target,
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
            for feat in feats[1:]
        )
        feat = self.fuse(torch.cat(ups, dim=1))
        heatmap = nn.functional.interpolate(
            self.head(feat),
            size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear",
            align_corners=DECODE_ALIGN_CORNERS,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat),
                size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear",
                align_corners=DECODE_ALIGN_CORNERS,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


def split_dataset(
    dataset: Dataset[Any], train_ratio: float, seed: int
) -> tuple[Subset[Any], Subset[Any]]:
    train_indices, val_indices = split_indices(len(dataset), train_ratio, seed)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_indices(n: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic shuffled train/val index split. Shared so two dataset
    instances over the same dir (augmented train, plain val) stay disjoint."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"--train_ratio must be between 0 and 1, got {train_ratio}")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_size = max(1, int(n * train_ratio))
    if n - train_size == 0:
        train_size = n - 1
    return indices[:train_size], indices[train_size:]


def build_augment_config(args: argparse.Namespace) -> AugmentConfig | None:
    """Return an AugmentConfig when --augment is set, else None (no augmentation)."""
    if not getattr(args, "augment", False):
        return None
    return AugmentConfig(
        hflip_prob=args.aug_hflip_prob,
        brightness=args.aug_brightness,
        contrast=args.aug_contrast,
        noise_std=args.aug_noise_std,
    )


def get_dataset_heatmap_size(dataset: CCTagHeatmapDataset) -> tuple[int, int]:
    if dataset.heatmap_height is None or dataset.heatmap_width is None:
        raise ValueError(
            f"Failed to infer heatmap size from dataset: {dataset.dataset_dir}"
        )
    return dataset.heatmap_height, dataset.heatmap_width


def build_dataset_collection(
    dataset_dirs: list[Path],
    input_size: tuple[int, int],
    split_name: str,
    augment: AugmentConfig | None = None,
) -> tuple[Dataset[Any], int, tuple[int, int]]:
    datasets = [
        CCTagHeatmapDataset(
            dataset_dir=dataset_dir, input_size=input_size, augment=augment
        )
        for dataset_dir in dataset_dirs
    ]
    if not datasets:
        raise ValueError(f"No datasets provided for {split_name}")

    heatmap_size = get_dataset_heatmap_size(datasets[0])
    for dataset in datasets[1:]:
        current_size = get_dataset_heatmap_size(dataset)
        if current_size != heatmap_size:
            raise ValueError(
                f"Incompatible heatmap size in {dataset.dataset_dir}: expected {heatmap_size}, got {current_size}"
            )

    if len(datasets) == 1:
        merged_dataset: Dataset[Any] = datasets[0]
    else:
        merged_dataset = ConcatDataset(datasets)
    return merged_dataset, sum(len(dataset) for dataset in datasets), heatmap_size


def resolve_dataset_sources(
    args: argparse.Namespace,
) -> tuple[list[Path] | None, list[Path] | None]:
    """Resolve explicit train/val dataset sources.

    Returns ``(train_dirs, val_dirs)`` where:
      - both None              -> fall back to the single --dataset_dir.
      - (train_dirs, val_dirs) -> explicit train and val sources.
      - (train_dirs, None)     -> one or more train sources; the val split is
                                  carved out of the merged training data using
                                  --train_ratio (seeded, disjoint).
    """
    train_dirs = list(args.train_dataset_dirs)
    val_dirs = list(args.val_dataset_dirs)

    if val_dirs and not train_dirs:
        raise ValueError(
            "--val_dataset_dir requires --train_dataset_dir. Provide training "
            "sources too, or use a single --dataset_dir with an internal split."
        )
    if train_dirs:
        return train_dirs, (val_dirs or None)

    return None, None


def create_dataloaders(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> tuple[
    DataLoader[Any],
    DataLoader[Any],
    int,
    tuple[int, int],
    DistributedSampler[Any] | None,
]:
    aug_cfg = build_augment_config(args)
    explicit_train_dirs, explicit_val_dirs = resolve_dataset_sources(args)
    if explicit_train_dirs is not None and explicit_val_dirs is not None:
        train_set, train_size, heatmap_size = build_dataset_collection(
            explicit_train_dirs,
            input_size=(args.input_width, args.input_height),
            split_name="train",
            augment=aug_cfg,
        )
        val_set, val_size, val_heatmap_size = build_dataset_collection(
            explicit_val_dirs,
            input_size=(args.input_width, args.input_height),
            split_name="val",
            augment=None,  # validation stays deterministic
        )
        if val_heatmap_size != heatmap_size:
            raise ValueError(
                f"Train/val heatmap size mismatch: train={heatmap_size}, val={val_heatmap_size}"
            )
        dataset_size = train_size + val_size
    elif explicit_train_dirs is not None:
        # One or more train sources, no explicit val: merge them and carve out a
        # val split with the seeded --train_ratio index split. Build two merged
        # collections over the same dirs so train can augment while val stays
        # deterministic; the shared seeded split keeps the two subsets disjoint.
        train_base, base_size, heatmap_size = build_dataset_collection(
            explicit_train_dirs,
            input_size=(args.input_width, args.input_height),
            split_name="train",
            augment=aug_cfg,
        )
        val_base, _, _ = build_dataset_collection(
            explicit_train_dirs,
            input_size=(args.input_width, args.input_height),
            split_name="val",
            augment=None,
        )
        train_idx, val_idx = split_indices(base_size, args.train_ratio, args.seed)
        train_set = Subset(train_base, train_idx)
        val_set = Subset(val_base, val_idx)
        train_size = len(train_set)
        val_size = len(val_set)
        dataset_size = base_size
    else:
        # Two instances over the same dir so the train split can augment while the
        # val split stays deterministic; the seeded index split keeps them disjoint.
        train_base = CCTagHeatmapDataset(
            dataset_dir=args.dataset_dir,
            input_size=(args.input_width, args.input_height),
            augment=aug_cfg,
        )
        val_base = CCTagHeatmapDataset(
            dataset_dir=args.dataset_dir,
            input_size=(args.input_width, args.input_height),
            augment=None,
        )
        train_idx, val_idx = split_indices(len(train_base), args.train_ratio, args.seed)
        train_set = Subset(train_base, train_idx)
        val_set = Subset(val_base, val_idx)
        train_size = len(train_set)
        val_size = len(val_set)
        dataset_size = len(train_base)
        heatmap_size = get_dataset_heatmap_size(train_base)

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
    return train_loader, val_loader, dataset_size, heatmap_size, train_sampler


def decode_heatmap_centers(
    heatmaps: torch.Tensor, offsets: torch.Tensor | None = None
) -> torch.Tensor:
    """Peak localization (batched, GPU-friendly).

    If `offsets` (B, 2, H, W) is given, decode as integer argmax + offset[peak] (CenterNet style).
    Otherwise fall back to weighted centroid around the argmax peak.
    """
    batch_size, _, height, width = heatmaps.shape
    hm = heatmaps.squeeze(1)

    flat_indices = hm.view(batch_size, -1).argmax(dim=1)
    py = torch.div(flat_indices, width, rounding_mode="floor")
    px = flat_indices % width

    if offsets is not None:
        b_idx = torch.arange(batch_size, device=hm.device)
        off = offsets[b_idx, :, py, px]  # (B, 2)
        results_x = px.float() + off[:, 0]
        results_y = py.float() + off[:, 1]
        return torch.stack((results_x, results_y), dim=1)

    radius = 2
    padded = nn.functional.pad(hm, (radius, radius, radius, radius), mode="replicate")
    results_x = torch.zeros(batch_size, device=hm.device)
    results_y = torch.zeros(batch_size, device=hm.device)
    for i in range(batch_size):
        cy, cx = int(py[i].item()), int(px[i].item())
        region = padded[i, cy : cy + 2 * radius + 1, cx : cx + 2 * radius + 1]
        total = region.sum()
        if total < 1e-12:
            results_x[i] = float(cx)
            results_y[i] = float(cy)
            continue
        yy = torch.arange(
            cy - radius, cy + radius + 1, device=hm.device, dtype=hm.dtype
        )
        xx = torch.arange(
            cx - radius, cx + radius + 1, device=hm.device, dtype=hm.dtype
        )
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        results_x[i] = (grid_x * region).sum() / total
        results_y[i] = (grid_y * region).sum() / total

    return torch.stack((results_x, results_y), dim=1)


def compute_center_distance_sum(
    prediction: torch.Tensor,
    target_centers_px: torch.Tensor,
    input_size: tuple[int, int],
    offsets: torch.Tensor | None = None,
    peak_threshold: float = 0.5,
) -> tuple[float, int]:
    # Only score localization on positives the model actually detected
    # (pred peak >= threshold). Undetected positives are false negatives and are
    # already reflected in positive_detection_rate; folding their argmax of a
    # near-zero heatmap into the L2 produces spurious hundreds-of-px errors that
    # dominate the mean (see eval_l2_distribution.py tail analysis).
    detected = prediction.flatten(1).max(dim=1).values >= peak_threshold
    if not detected.any():
        return 0.0, 0
    prediction = prediction[detected]
    target_centers_px = target_centers_px[detected]
    if offsets is not None:
        offsets = offsets[detected]
    pred_centers = decode_heatmap_centers(prediction, offsets=offsets)
    input_width, input_height = input_size
    heatmap_height, heatmap_width = prediction.shape[-2:]
    scale_x = input_width / float(heatmap_width)
    scale_y = input_height / float(heatmap_height)
    pred_centers_px = pred_centers.clone()
    pred_centers_px[:, 0] *= scale_x
    pred_centers_px[:, 1] *= scale_y
    distances = torch.linalg.norm(pred_centers_px - target_centers_px, dim=1)
    return float(distances.sum().detach().cpu().item()), int(distances.numel())


def compute_detection_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    peak_threshold: float = 0.5,
) -> tuple[int, int, int, int]:
    pred_peak = prediction.flatten(1).max(dim=1).values
    gt_peak = target.flatten(1).max(dim=1).values
    pred_has_object = pred_peak >= peak_threshold
    gt_has_object = gt_peak > 0.1

    tp = int(
        torch.logical_and(gt_has_object, pred_has_object).sum().detach().cpu().item()
    )
    fp = int(
        torch.logical_and(torch.logical_not(gt_has_object), pred_has_object)
        .sum()
        .detach()
        .cpu()
        .item()
    )
    fn = int(
        torch.logical_and(gt_has_object, torch.logical_not(pred_has_object))
        .sum()
        .detach()
        .cpu()
        .item()
    )
    tn = int(
        torch.logical_and(
            torch.logical_not(gt_has_object), torch.logical_not(pred_has_object)
        )
        .sum()
        .detach()
        .cpu()
        .item()
    )
    return tp, fp, fn, tn


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _split_pred(
    out: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Unpack model output into (heatmap, offset_or_None, size_or_None).

    Accepts either a bare heatmap tensor, a legacy 2-tuple (heatmap, offset),
    or the current 3-tuple (heatmap, offset_or_None, size_or_None).
    """
    if not isinstance(out, tuple):
        return out, None, None
    if len(out) == 2:
        return out[0], out[1], None
    return out[0], out[1], out[2]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: "torch.amp.GradScaler | None" = None,
    use_amp: bool = False,
    channels_last: bool = False,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = batch["heatmap"].to(device, non_blocking=True)
        heatmap_centers = batch["heatmap_center"].to(device, non_blocking=True)
        size_target = batch.get("size_target")
        if size_target is not None:
            size_target = size_target.to(device, non_blocking=True)
        occlusion_ratio = batch.get("occlusion_ratio")
        if occlusion_ratio is not None:
            occlusion_ratio = occlusion_ratio.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # autocast only wraps the forward pass; the loss is computed in fp32 because
        # BCELoss is unsafe under fp16 autocast. Cast the fp16 outputs back to fp32.
        with torch.autocast(device_type=device.type, enabled=use_amp):
            raw_out = model(images)
        predictions, offset_pred, size_pred = _split_pred(raw_out)
        predictions = predictions.float()
        offset_pred = offset_pred.float() if offset_pred is not None else None
        size_pred = size_pred.float() if size_pred is not None else None
        loss = criterion(
            predictions,
            targets,
            heatmap_centers=heatmap_centers,
            offset_pred=offset_pred,
            size_pred=size_pred,
            size_target=size_target if size_pred is not None else None,
            occlusion_ratio=occlusion_ratio,
        )
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    use_amp: bool = False,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_center_distance = 0.0
    sample_count = 0
    positive_count = 0
    tp_det = 0
    fp_det = 0
    fn_det = 0
    tn_det = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        targets = batch["heatmap"].to(device, non_blocking=True)
        centers = batch["center"].to(device, non_blocking=True)

        # Always run the validation forward in fp32: under fp16 autocast the sigmoid
        # peak saturates to a flat plateau of identical 1.0 cells, so argmax decode
        # snaps to the top-left of the plateau and inflates center_l2_px by ~10px.
        # Validation is infrequent, so the fp32 cost is negligible.
        raw_out = model(images)
        predictions, offset_pred, _size_pred = _split_pred(raw_out)
        predictions = predictions.float()
        offset_pred = offset_pred.float() if offset_pred is not None else None
        loss = criterion(predictions, targets)
        pos_mask = targets.flatten(1).max(dim=1).values > 0.1
        center_distance_sum = 0.0
        center_distance_count = 0
        if pos_mask.any():
            offsets_pos = offset_pred[pos_mask] if offset_pred is not None else None
            center_distance_sum, center_distance_count = compute_center_distance_sum(
                predictions[pos_mask],
                centers[pos_mask],
                input_size,
                offsets=offsets_pos,
            )
        batch_tp, batch_fp, batch_fn, batch_tn = compute_detection_counts(
            predictions, targets
        )

        batch_size = images.size(0)
        running_loss += tensor_to_float(loss) * batch_size
        running_center_distance += center_distance_sum
        sample_count += batch_size
        positive_count += center_distance_count
        tp_det += batch_tp
        fp_det += batch_fp
        fn_det += batch_fn
        tn_det += batch_tn

    total_loss = reduce_sum(running_loss, device)
    total_center_distance = reduce_sum(running_center_distance, device)
    total_count = reduce_sum(float(sample_count), device)
    total_positive_count = reduce_sum(float(positive_count), device)
    total_tp = reduce_sum(float(tp_det), device)
    total_fp = reduce_sum(float(fp_det), device)
    total_fn = reduce_sum(float(fn_det), device)
    total_tn = reduce_sum(float(tn_det), device)

    avg_loss = total_loss / max(total_count, 1.0)
    avg_center_error = total_center_distance / max(total_positive_count, 1.0)
    negative_false_positive_rate = safe_divide(total_fp, total_fp + total_tn)
    positive_detection_rate = safe_divide(total_tp, total_tp + total_fn)

    return {
        "val_loss": avg_loss,
        "center_l2_px_positive_only": avg_center_error,
        "negative_false_positive_rate": negative_false_positive_rate,
        "positive_detection_rate": positive_detection_rate,
    }


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip torch.compile (OptimizedModule._orig_mod) and DDP (.module) wrappers
    so saved state_dict keys match the raw model that inference loads."""
    inner = getattr(model, "_orig_mod", model)  # torch.compile wrapper
    if isinstance(inner, DistributedDataParallel):
        inner = inner.module
    return inner


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    world_size: int,
    heatmap_size: tuple[int, int],
) -> None:
    if not is_main_process():
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = _unwrap_model(model).state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                **vars(args),
                "world_size": world_size,
                "backbone": getattr(args, "backbone", "efficientnet_b0"),
                # Canonical head flags read by inference / benchmark.
                "use_offset_head": bool(getattr(args, "offset_head", False)),
                "use_size_head": bool(getattr(args, "size_head", False)),
                # Heatmap output resolution, so inference rebuilds the model at the
                # trained size instead of the (100, 160) fallback in load_model.
                "heatmap_height": heatmap_size[0],
                "heatmap_width": heatmap_size[1],
            },
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
        "train_dataset_dirs": [str(path) for path in args.train_dataset_dirs],
        "val_dataset_dirs": [str(path) for path in args.val_dataset_dirs],
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
        "uses_explicit_train_val_datasets": bool(
            args.train_dataset_dirs or args.val_dataset_dirs
        ),
        "seed": args.seed,
        "device": str(device),
        "world_size": world_size,
        "use_offset_head": bool(getattr(args, "offset_head", False)),
        "offset_weight": float(getattr(args, "offset_weight", 1.0)),
        "offset_head_hidden": int(getattr(args, "offset_head_hidden", 0)),
        "use_size_head": bool(getattr(args, "size_head", False)),
        "size_weight": float(getattr(args, "size_weight", 1.0)),
        "use_focal": bool(getattr(args, "focal_loss", False)),
        "backbone": getattr(args, "backbone", "efficientnet_b0"),
        "align_corners": bool(getattr(args, "align_corners", False)),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def append_metrics_row(
    metrics_path: Path,
    epoch: int,
    train_loss: float,
    val_metrics: dict[str, float],
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
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "center_l2_px_positive_only",
                    "negative_false_positive_rate",
                    "positive_detection_rate",
                    "lr",
                    "duration_sec",
                ]
            )
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{val_metrics['val_loss']:.6f}",
                f"{val_metrics['center_l2_px_positive_only']:.4f}",
                f"{val_metrics['negative_false_positive_rate']:.6f}",
                f"{val_metrics['positive_detection_rate']:.6f}",
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


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def build_model(
    heatmap_size: tuple[int, int],
    device: torch.device,
    rank: int,
    world_size: int,
    backbone: str = "efficientnet_b0",
    use_offset_head: bool = False,
    use_size_head: bool = False,
    offset_head_hidden: int = 0,
    channels_last: bool = False,
) -> nn.Module:
    # Rank 0 loads pretrained weights first so other ranks can reuse the local cache.
    if world_size > 1 and rank != 0:
        dist.barrier()
    head_kwargs = {
        "use_offset_head": use_offset_head,
        "use_size_head": use_size_head,
        "offset_head_hidden": offset_head_hidden,
    }
    if backbone == "mobilenet_v3_small":
        model = CCTagNetMobileV3(heatmap_size=heatmap_size, **head_kwargs).to(device)
    elif backbone == "resnet18":
        model = CCTagNetResNet18(heatmap_size=heatmap_size, **head_kwargs).to(device)
    elif backbone == "resnet18_hires":
        model = CCTagNetResNet18HiRes(heatmap_size=heatmap_size, **head_kwargs).to(
            device
        )
    elif backbone == "hrnet_w18":
        model = CCTagNetHRNet(
            heatmap_size=heatmap_size, variant=backbone, **head_kwargs
        ).to(device)
    else:
        model = CCTagNet(heatmap_size=heatmap_size, **head_kwargs).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if world_size > 1 and rank == 0:
        dist.barrier()
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            # offset_head/size_head only contribute to the loss when a batch
            # contains positive samples (see CombinedHeatmapLoss). An all-negative
            # batch on any rank leaves those params gradient-less, which deadlocks
            # the default reducer; let DDP detect and skip them per-iteration.
            find_unused_parameters=(use_offset_head or use_size_head),
        )
    return model


def main() -> None:
    args = parse_args()
    rank, world_size, _local_rank, device = setup_distributed(args)
    set_seed(args.seed + rank)

    # Apply the decoder align_corners ablation toggle before any model is built.
    global DECODE_ALIGN_CORNERS
    DECODE_ALIGN_CORNERS = bool(args.align_corners)

    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        # Fixed input size -> cudnn can autotune the fastest conv kernels.
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark
        # TF32 Tensor Core path for conv/matmul: 10-bit-mantissa multiply with
        # fp32 accumulate. Speeds up Ampere+ GPUs at accuracy that is effectively
        # lossless for CNN training (and the coordinate decode is elementwise, not
        # matmul, so it stays full fp32). Disable with --no-tf32 for strict repro.
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    train_loader, val_loader, dataset_size, heatmap_size, train_sampler = (
        create_dataloaders(
            args,
            rank=rank,
            world_size=world_size,
        )
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    preview_batch(train_loader)

    model = build_model(
        heatmap_size=heatmap_size,
        device=device,
        rank=rank,
        world_size=world_size,
        backbone=args.backbone,
        use_offset_head=args.offset_head,
        use_size_head=args.size_head,
        offset_head_hidden=args.offset_head_hidden,
        channels_last=args.channels_last,
    )
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        state_dict = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )
        target = model.module if isinstance(model, DistributedDataParallel) else model
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        if is_main_process():
            print(
                f"[resume] loaded weights from {args.resume_from} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
    if args.compile_model:
        # One-time graph capture + kernel fusion; fixed input size avoids recompiles.
        model = torch.compile(model)
        if is_main_process():
            print("[compile] torch.compile enabled")
    criterion = CombinedHeatmapLoss(
        use_focal=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        ohem_ratio=args.ohem_ratio,
        offset_weight=args.offset_weight,
        size_weight=args.size_weight,
        occ_weight_k=args.occ_loss_weight,
    ).to(device)
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    logger = (
        setup_logger(args.output_dir)
        if is_main_process()
        else logging.getLogger("train")
    )

    best_val_loss = math.inf
    if is_main_process():
        logger.info(
            f"training on {device} | world_size={world_size} | backbone={args.backbone} | "
            f"samples={dataset_size} train={train_size} val={val_size} "
            f"input={args.input_width}x{args.input_height} heatmap={heatmap_size[1]}x{heatmap_size[0]} "
            f"per_gpu_batch={args.batch_size} global_batch={args.batch_size * world_size}"
        )

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            use_amp=use_amp,
            channels_last=args.channels_last,
        )
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            input_size=(args.input_width, args.input_height),
            use_amp=use_amp,
            channels_last=args.channels_last,
        )
        scheduler.step()
        duration_sec = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        append_metrics_row(
            metrics_path=metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            val_metrics=val_metrics,
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
            heatmap_size=heatmap_size,
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
                heatmap_size=heatmap_size,
            )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                checkpoint_path=args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                args=args,
                world_size=world_size,
                heatmap_size=heatmap_size,
            )

        if is_main_process():
            logger.info(
                f"epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['val_loss']:.6f} "
                f"center_l2_px_positive_only={val_metrics['center_l2_px_positive_only']:.3f} "
                f"negative_false_positive_rate={val_metrics['negative_false_positive_rate']:.4f} "
                f"positive_detection_rate={val_metrics['positive_detection_rate']:.4f} "
                f"lr={current_lr:.6g} "
                f"time={duration_sec:.1f}s"
            )

    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if is_main_process():
            logging.getLogger("train").warning("training interrupted")
        cleanup_distributed()
        raise SystemExit(130)
