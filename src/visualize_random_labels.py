#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample images from a dataset and visualize center-point and ellipse labels."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("./outputs/visulaize/base_set"),
        help="Dataset root containing images/, labels_yolo/, and optional labels.csv.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of random images to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./outputs/tmp/random_label_check"),
        help="Directory to save the visualizations.",
    )
    parser.add_argument(
        "--show_yolo_bbox",
        action="store_true",
        help="Also draw the YOLO axis-aligned bounding box.",
    )
    return parser.parse_args()


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = int(round((x_center - width / 2.0) * image_width))
    y1 = int(round((y_center - height / 2.0) * image_height))
    x2 = int(round((x_center + width / 2.0) * image_width))
    y2 = int(round((y_center + height / 2.0) * image_height))
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))
    return x1, y1, x2, y2


def load_csv_labels(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row for row in reader if row.get("filename")}


def draw_fitted_ellipse(image: np.ndarray, row: dict[str, str]) -> None:
    required = [
        "ellipse_cx",
        "ellipse_cy",
        "ellipse_a",
        "ellipse_b",
        "ellipse_angle_rad",
    ]
    if not all(row.get(key) for key in required):
        return

    cx = float(row["ellipse_cx"])
    cy = float(row["ellipse_cy"])
    axis_a = max(int(round(float(row["ellipse_a"]))), 1)
    axis_b = max(int(round(float(row["ellipse_b"]))), 1)
    angle_rad = float(row["ellipse_angle_rad"])
    angle_deg = float(np.degrees(angle_rad))
    center = (int(round(cx)), int(round(cy)))

    cv2.ellipse(
        image,
        center,
        (axis_a, axis_b),
        angle_deg,
        0,
        360,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        image,
        center,
        (255, 200, 0),
        cv2.MARKER_TILTED_CROSS,
        12,
        2,
        cv2.LINE_AA,
    )

    major_dx = int(round(np.cos(angle_rad) * axis_a))
    major_dy = int(round(np.sin(angle_rad) * axis_a))
    minor_dx = int(round(-np.sin(angle_rad) * axis_b))
    minor_dy = int(round(np.cos(angle_rad) * axis_b))

    cv2.line(
        image,
        (center[0] - major_dx, center[1] - major_dy),
        (center[0] + major_dx, center[1] + major_dy),
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        (center[0] - minor_dx, center[1] - minor_dy),
        (center[0] + minor_dx, center[1] + minor_dy),
        (255, 120, 0),
        2,
        cv2.LINE_AA,
    )


def draw_center_point(image: np.ndarray, row: dict[str, str]) -> None:
    x_value = row.get("x") or row.get("center_x")
    y_value = row.get("y") or row.get("center_y")
    if not x_value or not y_value:
        return

    cx = int(round(float(x_value)))
    cy = int(round(float(y_value)))

    cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)
    cv2.circle(image, (cx, cy), 8, (0, 0, 0), 2)


def draw_label_legend(
    image: np.ndarray,
    row: dict[str, str] | None,
    class_id: str | None,
    show_yolo_bbox: bool,
) -> None:
    text_lines = []
    if show_yolo_bbox and class_id is not None:
        text_lines.append(f"YOLO class {class_id}")
    if row:
        x_value = row.get("x") or row.get("center_x")
        y_value = row.get("y") or row.get("center_y")
        if x_value and y_value:
            text_lines.append(f"x,y=({float(x_value):.1f},{float(y_value):.1f})")
        if row.get("ellipse_cx") and row.get("ellipse_cy"):
            text_lines.append(
                f"ellipse_c=({float(row['ellipse_cx']):.1f},{float(row['ellipse_cy']):.1f})"
            )
        if row.get("ellipse_a") and row.get("ellipse_b"):
            text_lines.append(
                f"ellipse a,b=({float(row['ellipse_a']):.1f},{float(row['ellipse_b']):.1f})"
            )
        if row.get("ellipse_angle_rad"):
            text_lines.append(f"angle={float(row['ellipse_angle_rad']):.2f} rad")

    if not text_lines:
        return

    line_height = 18
    box_height = 8 + line_height * len(text_lines)
    cv2.rectangle(image, (0, 28), (image.shape[1], 28 + box_height), (32, 32, 32), -1)
    for idx, text in enumerate(text_lines):
        cv2.putText(
            image,
            text,
            (8, 28 + 16 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_labels(
    image_path: Path,
    label_path: Path,
    heatmap_path: Path | None,
    csv_row: dict[str, str] | None,
    show_yolo_bbox: bool,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = image.shape[:2]
    overlay = image.copy()

    if heatmap_path and heatmap_path.exists():
        if heatmap_path.suffix == ".npy":
            heatmap_raw = np.load(heatmap_path).astype(np.float32)
            # Resize from strided shape to original image shape
            heatmap_resized = cv2.resize(
                heatmap_raw, (w, h), interpolation=cv2.INTER_LINEAR
            )
            heatmap_u8 = np.clip(heatmap_resized * 255, 0, 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)
        else:
            heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
            if heatmap is not None:
                heatmap_resized = cv2.resize(
                    heatmap, (w, h), interpolation=cv2.INTER_LINEAR
                )
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)

    class_id = None

    if show_yolo_bbox and label_path.exists():
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = parts[0]
            x_center, y_center, box_w, box_h = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, box_w, box_h, w, h)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"class {class_id}"
            label_y = y1 - 8 if y1 > 24 else y1 + 24
            cv2.putText(
                overlay,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
    elif show_yolo_bbox:
        cv2.putText(
            overlay,
            "missing label",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if csv_row:
        draw_center_point(overlay, csv_row)
        draw_fitted_ellipse(overlay, csv_row)

    draw_label_legend(overlay, csv_row, class_id, show_yolo_bbox)
    return overlay


def main() -> None:
    args = parse_args()
    image_dir = args.dataset_dir / "images"
    label_dir = args.dataset_dir / "labels_yolo"
    heatmap_dir = args.dataset_dir / "heatmaps"
    csv_rows = load_csv_labels(args.dataset_dir / "labels.csv")

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    rng = random.Random(args.seed)
    sample_count = min(args.num_samples, len(image_paths))
    sampled_images = rng.sample(image_paths, sample_count)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving visualizations to: {args.output_dir}")
    print("Sampled files:")
    for image_path in sampled_images:
        label_path = label_dir / f"{image_path.stem}.txt"

        heatmap_path = heatmap_dir / f"{image_path.stem}.npy"
        if not heatmap_path.exists():
            heatmap_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                hp = heatmap_dir / f"{image_path.stem}{ext}"
                if hp.exists():
                    heatmap_path = hp
                    break

        # Handle fallback for csv reading just in case stem vs name issue exists
        csv_row = csv_rows.get(image_path.name) or csv_rows.get(image_path.stem)

        visualized = draw_labels(
            image_path,
            label_path,
            heatmap_path,
            csv_row,
            args.show_yolo_bbox,
        )

        out_path = args.output_dir / f"{image_path.stem}_vis.jpg"
        success = cv2.imwrite(str(out_path), visualized)
        if not success:
            raise ValueError(f"Failed to write output image: {out_path}")

        print(f"  - {out_path.name}")


if __name__ == "__main__":
    main()
