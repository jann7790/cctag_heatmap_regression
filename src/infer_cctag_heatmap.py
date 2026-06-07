#!/usr/bin/env python3
"""
Inference script for CCTagNet heatmap model.

Example:
  # single image
  python src/infer_cctag_heatmap.py --checkpoint ./outputs/runs/experiment_01/best.pt --input ./assets/samples/cctag_reallife.png

  # directory of images (saves heatmaps and prints centers)
  python src/infer_cctag_heatmap.py --checkpoint ./outputs/runs/experiment_01/best.pt --input ./outputs/testing/small_testing/images --output ./outputs/inference/results_demo
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path, PosixPath
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


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
    parser.add_argument("--compile", action="store_true", help="Use torch.compile() for faster inference (slow first run).")
    parser.add_argument("--amp", action="store_true", help="(Deprecated/ignored) fp16 corrupts center localization via peak-plateau argmax bias; inference always decodes in fp32.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference (default: 1).")
    parser.add_argument("--decode_method", type=str, default="weighted",
                        choices=["weighted", "subpixel", "argmax"],
                        help="Peak localization method: 'weighted' (weighted centroid, default), "
                             "'subpixel' (Hessian refinement), 'argmax' (integer grid).")
    parser.add_argument("--min_peak_sharpness", type=float, default=0.0,
                        help="Minimum peak sharpness ratio (peak / mean of top region) to accept a detection. "
                             "Real CCTags produce sharp peaks (ratio > 3); false positives from overexposure "
                             "produce diffuse activations (ratio < 2). Set to 0 to disable. Recommended: 3.0.")
    parser.add_argument("--tracking_mode", action="store_true",
                        help="Enable conservative tracking defaults: threshold=0.65, min_peak_sharpness=3.0.")
    parser.add_argument("--temporal_window", type=int, default=0,
                        help="Require detection in at least ceil(N*0.6) of the last N frames to accept. "
                             "Filters out sporadic single-frame false positives. 0 disables. Recommended: 5.")
    parser.add_argument("--worst_n", type=int, default=20,
                        help="In --eval mode, report the N worst-localized positives (highest center L2). "
                             "Saved to worst_center_errors.csv. Default: 20.")
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


def ensure_mobilenet_v3_small():
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        raise SystemExit("pip install torchvision") from exc
    return mobilenet_v3_small, MobileNet_V3_Small_Weights


def ensure_resnet18():
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise SystemExit("pip install torchvision") from exc
    return resnet18, ResNet18_Weights


class CCTagNet(nn.Module):
    def __init__(self, heatmap_size: tuple[int, int], use_offset_head: bool = False,
                 use_size_head: bool = False) -> None:
        super().__init__()
        efficientnet_b0, EfficientNet_B0_Weights = ensure_torchvision()
        backbone = efficientnet_b0(weights=None)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.use_offset_head = use_offset_head
        self.use_size_head = use_size_head
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
        # NOTE: legacy CCTagNet has no separate feature tap; offset_head / size_head are unsupported here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pred = self.decoder(features)
        return nn.functional.interpolate(
            pred, size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear", align_corners=False,
        )


class CCTagNetV3(nn.Module):
    """U-Net style CCTagNet with skip connections (used by experiment_mixed_v3)."""

    SKIP_INDICES = (1, 2, 3, 4)

    def __init__(self, heatmap_size: tuple[int, int]) -> None:
        super().__init__()
        efficientnet_b0, EfficientNet_B0_Weights = ensure_torchvision()
        backbone = efficientnet_b0(weights=None)
        self.encoder = backbone.features
        self.heatmap_height, self.heatmap_width = heatmap_size
        self.dec1 = self._dec_block(1280 + 80, 256)
        self.dec2 = self._dec_block(256 + 40, 128)
        self.dec3 = self._dec_block(128 + 24, 64)
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
            x, size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear", align_corners=False,
        )


class CCTagNetMobileV3(nn.Module):
    """MobileNetV3-Small encoder with U-Net style skip connections."""

    SKIP_INDICES = (0, 1, 3, 8)
    BOTTLENECK_CHANNELS = 576

    def __init__(self, heatmap_size: tuple[int, int], use_offset_head: bool = False,
                 use_size_head: bool = False) -> None:
        super().__init__()
        mobilenet_v3_small, _ = ensure_mobilenet_v3_small()
        backbone = mobilenet_v3_small(weights=None)
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
            self.offset_head = nn.Conv2d(16, 2, kernel_size=1)
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
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
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
            self.head(feat), size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear", align_corners=False,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat), size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear", align_corners=False,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat), size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear", align_corners=False,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    try:
        with torch.serialization.safe_globals([PosixPath]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        # PyTorch 2.6 defaults to weights_only=True, but older checkpoints may
        # embed trusted Python objects such as pathlib.PosixPath in metadata.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    heatmap_size = (config.get("heatmap_height", 100), config.get("heatmap_width", 160))
    backbone = config.get("backbone", "efficientnet_b0")
    state = ckpt["model_state_dict"]
    use_offset_head = bool(config.get("use_offset_head", False)) or any(k.startswith("offset_head.") for k in state)
    use_size_head = bool(config.get("use_size_head", False)) or any(k.startswith("size_head.") for k in state)
    config["use_offset_head"] = use_offset_head
    config["use_size_head"] = use_size_head
    head_kwargs = {"use_offset_head": use_offset_head, "use_size_head": use_size_head}
    if backbone == "mobilenet_v3_small":
        model = CCTagNetMobileV3(heatmap_size=heatmap_size, **head_kwargs).to(device)
    elif backbone == "resnet18":
        model = CCTagNetResNet18(heatmap_size=heatmap_size, **head_kwargs).to(device)
    elif "dec1.0.weight" in state:
        model = CCTagNetV3(heatmap_size=heatmap_size).to(device)
    else:
        model = CCTagNet(heatmap_size=heatmap_size, **head_kwargs).to(device)
    model.load_state_dict(state)
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


def decode_center_offset(heatmap: np.ndarray, offset: np.ndarray, threshold: float = 0.3) -> tuple[float, float] | None:
    """Argmax peak + predicted (dx, dy) offset at that pixel (CenterNet style)."""
    peak_val = float(heatmap.max())
    if peak_val < threshold:
        return None
    h, w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, w)
    dx = float(offset[0, py, px])
    dy = float(offset[1, py, px])
    return float(px) + dx, float(py) + dy


def decode_size_at_peak(
    heatmap: np.ndarray, size_log: np.ndarray, threshold: float = 0.3
) -> tuple[float, float] | None:
    """Read (ellipse_a, ellipse_b) in source-image pixels from the size head.

    The head outputs log(a), log(b); this returns exp of the prediction at the
    heatmap argmax peak. Returns None when the peak is below threshold.
    """
    peak_val = float(heatmap.max())
    if peak_val < threshold:
        return None
    h, w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, w)
    a_log = float(size_log[0, py, px])
    b_log = float(size_log[1, py, px])
    return float(np.exp(a_log)), float(np.exp(b_log))


def decode_center_weighted(heatmap: np.ndarray, threshold: float = 0.3, radius: int = 2) -> tuple[float, float] | None:
    """Return (x, y) via weighted centroid around the peak. More robust than Hessian for smooth Gaussians."""
    peak_val = float(heatmap.max())
    if peak_val < threshold:
        return None

    h, w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, w)

    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)

    region = heatmap[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = region.sum()
    if total < 1e-12:
        return float(px), float(py)
    cx = float((xx * region).sum() / total)
    cy = float((yy * region).sum() / total)
    return cx, cy


def compute_peak_sharpness(heatmap: np.ndarray, radius: int = 5) -> float:
    """Compute peak sharpness as peak_value / mean_of_surrounding_region.

    A sharp Gaussian peak (real CCTag) has a high ratio (> 3).
    A diffuse broad activation (false positive from overexposure) has a low ratio (< 2).
    """
    peak_val = float(heatmap.max())
    if peak_val < 1e-6:
        return 0.0
    h, w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    py, px = divmod(flat_idx, w)
    y_lo = max(0, py - radius)
    y_hi = min(h, py + radius + 1)
    x_lo = max(0, px - radius)
    x_hi = min(w, px + radius + 1)
    region = heatmap[y_lo:y_hi, x_lo:x_hi]
    region_mean = float(region.mean())
    if region_mean < 1e-6:
        return 0.0
    return peak_val / region_mean


class _TemporalFilter:
    """Ring buffer that requires M-of-N recent frames to have detections."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.min_hits = int(np.ceil(window * 0.6))
        self.buffer: list[bool] = []

    def update(self, detected: bool) -> bool:
        self.buffer.append(detected)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return sum(self.buffer) >= self.min_hits


class CCTagNetResNet18(nn.Module):
    """ResNet-18 encoder with U-Net style skip connections.

    Skip taps:
        relu(conv1) → 64ch, stride 2
        layer1      → 64ch, stride 4
        layer2      → 128ch, stride 8
        layer3      → 256ch, stride 16
    Bottleneck (layer4) → 512ch, stride 32
    """

    def __init__(self, heatmap_size: tuple[int, int], use_offset_head: bool = False,
                 use_size_head: bool = False) -> None:
        super().__init__()
        resnet18, ResNet18_Weights = ensure_resnet18()
        backbone = resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
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
            self.offset_head = nn.Conv2d(32, 2, kernel_size=1)
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
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
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
            self.head(feat), size=(self.heatmap_height, self.heatmap_width),
            mode="bilinear", align_corners=False,
        )
        offset = None
        if self.use_offset_head:
            offset = nn.functional.interpolate(
                self.offset_head(feat), size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear", align_corners=False,
            )
        size = None
        if self.use_size_head:
            size = nn.functional.interpolate(
                self.size_head(feat), size=(self.heatmap_height, self.heatmap_width),
                mode="bilinear", align_corners=False,
            )
        if self.use_offset_head or self.use_size_head:
            return heatmap, offset, size
        return heatmap


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
                "heatmap_path": resolve_heatmap_path(heatmaps_dir, filename),
                "center_x": float(row.get("center_x") or row.get("x") or -1.0),
                "center_y": float(row.get("center_y") or row.get("y") or -1.0),
                "is_negative": int(row.get("is_negative") or 0) == 1,
                "has_visible_marker": int(row.get("has_visible_marker") or 0) == 1,
                "ellipse_a": float(row.get("ellipse_a") or 0.0),
                "ellipse_b": float(row.get("ellipse_b") or 0.0),
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
    worst_n: int = 20,
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

    center_rows = [row for row in rows if row["center_l2_px"] is not None]
    center_errors = np.array([row["center_l2_px"] for row in center_rows], dtype=np.float64)
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
            "mean_center_l2_px_on_detected_positives": float(center_errors.mean()) if center_errors.size else None,
        }
    )
    if center_errors.size:
        metrics.update(
            {
                "center_l2_px_count": int(center_errors.size),
                "center_l2_px_median": float(np.median(center_errors)),
                "center_l2_px_p90": float(np.percentile(center_errors, 90)),
                "center_l2_px_p95": float(np.percentile(center_errors, 95)),
                "center_l2_px_max": float(center_errors.max()),
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

    # ── worst-localized positives (highest center L2) ─────────────────────────
    worst_rows = sorted(center_rows, key=lambda r: r["center_l2_px"], reverse=True)[: max(worst_n, 0)]
    if worst_rows:
        print(f"\n=== worst {len(worst_rows)} center localizations (px) ===")
        for r in worst_rows:
            print(f"  {r['center_l2_px']:8.2f}px  peak={r['peak']:.3f}  {r['filename']}")

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
        if worst_rows:
            worst_path = output_dir / "worst_center_errors.csv"
            with worst_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["filename", "center_l2_px", "peak"]
                )
                writer.writeheader()
                writer.writerows(
                    {"filename": r["filename"], "center_l2_px": r["center_l2_px"], "peak": r["peak"]}
                    for r in worst_rows
                )
            print(f"saved worst center errors: {worst_path}")


def save_overlay(
    image_path: Path,
    heatmap: np.ndarray,
    center_xy: tuple[float, float],
    out_path: Path,
    size_ab: tuple[float, float] | None = None,
) -> None:
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
    if size_ab is not None:
        a_px, b_px = size_ab
        if a_px > 0 and b_px > 0:
            axes = (max(int(round(a_px)), 1), max(int(round(b_px)), 1))
            cv2.ellipse(overlay, (cx, cy), axes, 0.0, 0.0, 360.0, (0, 255, 255), 2)
    cv2.circle(overlay, (cx, cy), 6, (0, 255, 0), -1)
    cv2.circle(overlay, (cx, cy), 8, (0, 0, 0),   2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Apply conservative tracking defaults when --tracking_mode is set
    if args.tracking_mode:
        if args.threshold == 0.5:  # only override if user didn't set explicitly
            args.threshold = 0.65
        if args.min_peak_sharpness == 0.0:
            args.min_peak_sharpness = 3.0
        if args.temporal_window == 0:
            args.temporal_window = 5

    temporal_filter: _TemporalFilter | None = None
    if args.temporal_window > 0:
        temporal_filter = _TemporalFilter(args.temporal_window)

    device = resolve_device(args.device)

    print(f"loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)

    if args.compile:
        print("compiling model with torch.compile (first run will be slow)...")
        model = torch.compile(model)

    input_width  = config.get("input_width",  640)
    input_height = config.get("input_height", 400)
    backbone = config.get("backbone", "efficientnet_b0")
    print(f"model input: {input_width}x{input_height}  backbone: {backbone}  device: {device}")

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

    # fp16 autocast saturates the sigmoid peak into a flat plateau of identical 1.0
    # cells; argmax then snaps to the plateau's top-left corner, biasing the decoded
    # center toward the origin by ~14px (detection rate is unaffected, so it hides
    # easily). Localization must be decoded in fp32, so --amp is ignored here.
    if args.amp and device.type == "cuda":
        print("warning: --amp ignored for inference — fp16 corrupts center "
              "localization (peak-plateau argmax bias). Running forward in fp32.")
    use_amp = False

    detection_log: list[dict[str, Any]] = []

    def _process_single(img_path: Path, heatmap: np.ndarray, heatmap_tensor: torch.Tensor,
                        offset: np.ndarray | None = None,
                        size_log: np.ndarray | None = None) -> None:
        """Post-process a single image: print result, save outputs, evaluate."""
        peak_val = float(heatmap.max())
        if offset is not None:
            result = decode_center_offset(heatmap, offset, threshold=args.threshold)
        elif args.decode_method == "weighted":
            result = decode_center_weighted(heatmap, threshold=args.threshold)
        elif args.decode_method == "subpixel":
            result = decode_center_subpixel(heatmap, threshold=args.threshold)
        else:
            result = decode_center(heatmap, threshold=args.threshold)

        size_ab: tuple[float, float] | None = None
        if size_log is not None:
            size_ab = decode_size_at_peak(heatmap, size_log, threshold=args.threshold)

        # Peak sharpness filter: reject diffuse broad activations
        sharpness = 0.0
        if result is not None and args.min_peak_sharpness > 0:
            sharpness = compute_peak_sharpness(heatmap)
            if sharpness < args.min_peak_sharpness:
                print(f"{img_path.name}  REJECTED (sharpness={sharpness:.2f} < {args.min_peak_sharpness})  peak={peak_val:.3f}")
                result = None

        # Temporal consistency filter: require M-of-N recent detections
        if temporal_filter is not None:
            accepted = temporal_filter.update(result is not None)
            if result is not None and not accepted:
                print(f"{img_path.name}  REJECTED (temporal filter: insufficient consecutive detections)  peak={peak_val:.3f}")
                result = None

        cx_px, cy_px = 0.0, 0.0
        if result is None:
            print(f"{img_path.name}  NO DETECTION  peak={peak_val:.3f} (< {args.threshold})")
            detection_log.append({
                "filename": img_path.name,
                "detected": False,
                "peak": peak_val,
                "center_x_px": None,
                "center_y_px": None,
                "ellipse_a_px": None,
                "ellipse_b_px": None,
            })
        else:
            cx_hm, cy_hm = result
            hm_h, hm_w = heatmap.shape
            tensor_for_orig, (orig_w, orig_h) = _orig_sizes[img_path]
            cx_px = cx_hm * orig_w / hm_w
            cy_px = cy_hm * orig_h / hm_h
            sharpness_str = f"  sharpness={sharpness:.2f}" if args.min_peak_sharpness > 0 else ""
            size_str = f"  ellipse=({size_ab[0]:.1f}, {size_ab[1]:.1f})px" if size_ab is not None else ""
            print(
                f"{img_path.name}  center=({cx_px:.1f}, {cy_px:.1f})px  "
                f"heatmap_peak=({cx_hm:.1f}, {cy_hm:.1f})  peak={peak_val:.3f}"
                f"{sharpness_str}{size_str}"
            )
            detection_log.append({
                "filename": img_path.name,
                "detected": True,
                "peak": peak_val,
                "center_x_px": cx_px,
                "center_y_px": cy_px,
                "ellipse_a_px": size_ab[0] if size_ab is not None else None,
                "ellipse_b_px": size_ab[1] if size_ab is not None else None,
            })

        if args.output:
            stem = img_path.stem
            if args.vis and result is not None:
                save_overlay(img_path, heatmap, result, args.output / f"{stem}_overlay.png", size_ab=size_ab)

        if gt_map is not None:
            stem = img_path.stem
            gt_row = gt_map.get(stem)
            if gt_row is None:
                print(f"{img_path.name}  warning: missing GT row in labels.csv, skipped evaluation")
                return

            gt_heatmap_path = gt_row["heatmap_path"]
            if not gt_heatmap_path.is_file():
                print(f"{img_path.name}  warning: missing GT heatmap {gt_heatmap_path}, skipped evaluation")
                return

            gt_heatmap_raw = load_heatmap(gt_heatmap_path)
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

    # Cache original sizes for batch processing
    _orig_sizes: dict[Path, tuple[None, tuple[int, int]]] = {}

    t_start = time.perf_counter()
    with torch.inference_mode():
        # Process in batches
        for batch_start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[batch_start : batch_start + args.batch_size]
            batch_tensors = []
            for img_path in batch_paths:
                tensor, orig_size = preprocess(img_path, input_width, input_height, device)
                batch_tensors.append(tensor)
                _orig_sizes[img_path] = (None, orig_size)

            batch_input = torch.cat(batch_tensors, dim=0)  # (B, 3, H, W)

            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    batch_out = model(batch_input)
            else:
                batch_out = model(batch_input)
            batch_offsets = None
            batch_sizes = None
            if isinstance(batch_out, tuple):
                if len(batch_out) == 2:
                    batch_heatmaps, batch_offsets = batch_out
                else:
                    batch_heatmaps, batch_offsets, batch_sizes = batch_out
            else:
                batch_heatmaps = batch_out

            for i, img_path in enumerate(batch_paths):
                heatmap_tensor = batch_heatmaps[i : i + 1]  # (1, 1, H, W)
                heatmap = heatmap_tensor[0, 0].float().cpu().numpy()
                offset_np: np.ndarray | None = None
                if batch_offsets is not None:
                    offset_np = batch_offsets[i].float().cpu().numpy()  # (2, H, W)
                size_np: np.ndarray | None = None
                if batch_sizes is not None:
                    size_np = batch_sizes[i].float().cpu().numpy()  # (2, H, W) log-space
                _process_single(img_path, heatmap, heatmap_tensor.float(), offset_np, size_np)

    t_elapsed = time.perf_counter() - t_start
    n_images = len(image_paths)
    print(f"\ninference: {n_images} images in {t_elapsed:.2f}s ({t_elapsed/n_images*1000:.1f}ms/image, {n_images/t_elapsed:.1f} FPS)")
    if use_amp:
        print("  (AMP float16 enabled)")
    if args.compile:
        print("  (torch.compile enabled — first-run overhead included)")

    # ── Detection summary (always printed) ────────────────────────────────────
    detected = [d for d in detection_log if d["detected"]]
    not_detected = [d for d in detection_log if not d["detected"]]
    total = len(detection_log)
    print("\n=== detection summary ===")
    print(f"total images:   {total}")
    print(f"detected:       {len(detected)}  ({len(detected)/total*100:.1f}%)" if total else "detected: 0")
    print(f"not detected:   {len(not_detected)}  ({len(not_detected)/total*100:.1f}%)" if total else "not detected: 0")

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        detected_path = args.output / "detected_images.txt"
        not_detected_path = args.output / "not_detected_images.txt"
        detection_csv = args.output / "detection_log.csv"
        detected_path.write_text("\n".join(d["filename"] for d in detected) + "\n", encoding="utf-8")
        not_detected_path.write_text("\n".join(d["filename"] for d in not_detected) + "\n", encoding="utf-8")
        with detection_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filename", "detected", "peak", "center_x_px", "center_y_px",
                    "ellipse_a_px", "ellipse_b_px",
                ],
            )
            writer.writeheader()
            writer.writerows(detection_log)
        print(f"saved detected list:     {detected_path}")
        print(f"saved not-detected list: {not_detected_path}")
        print(f"saved detection log:     {detection_csv}")

    if gt_map is not None:
        summarize_evaluation(eval_rows, args.output, worst_n=args.worst_n)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
