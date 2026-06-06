#!/usr/bin/env python3
"""
Convert one or more generated CCTag datasets into the Ultralytics YOLO layout.

The generator already emits YOLO-format labels:
  DATASET/images/<name>.png
  DATASET/labels_yolo/<name>.txt   # "<cls> cx cy w h" for positives; EMPTY for negatives

Ultralytics locates labels by swapping "/images/" -> "/labels/" in each image path,
so this builds:
  OUTPUT/images/train/*.png   OUTPUT/labels/train/*.txt
  OUTPUT/images/val/*.png     OUTPUT/labels/val/*.txt
  OUTPUT/data.yaml

Empty label files are preserved (Ultralytics treats them as background negatives),
which is exactly what the precision-leaning detector wants.

Example:
  python src/prepare_yolo_dataset.py \
    --dataset_dir ./outputs/training_sets/detection_sets/det_positive_wide \
    --dataset_dir ./outputs/training_sets/detection_sets/det_hard_negative \
    --output_dir ./outputs/datasets/yolo_detection --train_ratio 0.9
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Ultralytics YOLO dataset from generated CCTag datasets."
    )
    parser.add_argument(
        "--dataset_dir",
        dest="dataset_dirs",
        action="append",
        type=Path,
        default=[],
        required=False,
        help="Source dataset root (contains images/ and labels_yolo/). Repeat to merge sources.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination root for the Ultralytics layout (images/, labels/, data.yaml).",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio (default: 0.9).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the split.")
    parser.add_argument("--class_name", type=str, default="cctag", help="Single class name written into data.yaml.")
    parser.add_argument(
        "--link_mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy"],
        help="How to place files into the layout (default: symlink).",
    )
    args = parser.parse_args()
    if not args.dataset_dirs:
        parser.error("Provide at least one --dataset_dir.")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error(f"--train_ratio must be between 0 and 1, got {args.train_ratio}")
    return args


def collect_samples(dataset_dir: Path) -> list[tuple[Path, Path | None]]:
    """Return (image_path, label_path_or_None) pairs for a dataset root."""
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels_yolo"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    samples: list[tuple[Path, Path | None]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        samples.append((image_path, label_path if label_path.is_file() else None))
    if not samples:
        raise ValueError(f"No images found under {images_dir}")
    return samples


def place_file(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        # Absolute target so the link resolves regardless of CWD.
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def write_label(label_src: Path | None, dst: Path, link_mode: str) -> None:
    """Place the label file; synthesize an empty one when the source is a negative
    with no .txt (Ultralytics reads a missing/empty label as a background image)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if label_src is None:
        dst.write_text("", encoding="utf-8")
    else:
        place_file(label_src, dst, link_mode)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Clear any previously generated layout so each run is a clean, deterministic
    # build. Without this, a prior run that ordered --dataset_dir differently leaves
    # stale files under different s{idx}_ prefixes -- duplicating sources and leaking
    # the same image across the train/val split. Only the auto-generated images/ and
    # labels/ subtrees are removed; the real source datasets live elsewhere.
    for sub in ("images", "labels"):
        stale = args.output_dir / sub
        if stale.exists():
            shutil.rmtree(stale)
            print(f"cleared previous layout: {stale}")

    # Gather all samples from every source, with a per-source prefix to avoid name
    # collisions when multiple datasets share the 000000-style numbering.
    all_samples: list[tuple[str, Path, Path | None]] = []
    for idx, dataset_dir in enumerate(args.dataset_dirs):
        prefix = f"s{idx}_{dataset_dir.name}"
        for image_path, label_path in collect_samples(dataset_dir):
            unique = f"{prefix}__{image_path.stem}"
            all_samples.append((unique, image_path, label_path))

    rng.shuffle(all_samples)
    n_train = max(1, int(len(all_samples) * args.train_ratio))
    if n_train >= len(all_samples):
        n_train = len(all_samples) - 1
    splits = {"train": all_samples[:n_train], "val": all_samples[n_train:]}

    out = args.output_dir
    n_pos = n_neg = 0
    for split, samples in splits.items():
        for unique, image_path, label_path in samples:
            place_file(image_path, out / "images" / split / f"{unique}{image_path.suffix}", args.link_mode)
            write_label(label_path, out / "labels" / split / f"{unique}.txt", args.link_mode)
            if label_path is not None and label_path.stat().st_size > 0:
                n_pos += 1
            else:
                n_neg += 1

    data_yaml = out / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        "# Auto-generated by src/prepare_yolo_dataset.py\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        f"names: ['{args.class_name}']\n",
        encoding="utf-8",
    )

    print(f"sources         : {[str(d) for d in args.dataset_dirs]}")
    print(f"total samples   : {len(all_samples)}  (train={len(splits['train'])}, val={len(splits['val'])})")
    print(f"positives/neg   : {n_pos} positive, {n_neg} background")
    print(f"layout          : {out}/images/{{train,val}} + labels/{{train,val}}")
    print(f"data.yaml       : {data_yaml}")


if __name__ == "__main__":
    main()
