#!/usr/bin/env python3
"""Audit and materialize the fixed-tile no-occlusion training dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BIN_EDGES = (48.0, 80.0, 128.0, 192.0, 320.0)
BIN_WEIGHTS = (0.30, 0.30, 0.20, 0.20)
FIXED_POSITIVE_COUNTS = (4500, 4500, 3000, 3000)
BOUNDARY_FRACTION = 0.10
MIN_VISIBLE = 0.25
FULL_VISIBLE = 0.999


@dataclass(frozen=True)
class SourceRow:
    source: Path
    row: dict[str, str]


def _float(row: dict[str, str], key: str, default: float) -> float:
    value = row.get(key)
    return default if value is None or value == "" else float(value)


def diameter_px(row: dict[str, str]) -> float:
    return 2.0 * max(_float(row, "ellipse_a", 0.0), _float(row, "ellipse_b", 0.0))


def scale_bin(row: dict[str, str]) -> int | None:
    diameter = diameter_px(row)
    for index, (low, high) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        if low <= diameter < high or (index == len(BIN_WEIGHTS) - 1 and diameter == high):
            return index
    return None


def row_mode(row: dict[str, str], min_visible: float = MIN_VISIBLE) -> str:
    if int(row.get("is_negative") or 0):
        return "negative"
    visible = _float(row, "visible_marker_ratio", 1.0)
    clamped = int(row.get("target_clamped") or 0)
    occ = _float(row, "occlusion_ratio", 0.0)
    if occ > 0.0:
        return "rejected_occluded"
    if visible < min_visible:
        return "rejected_low_visibility"
    if min_visible <= visible < FULL_VISIBLE and clamped == 1:
        return "boundary"
    if visible >= FULL_VISIBLE and clamped == 0:
        return "full"
    return "rejected_invalid_target"


def source_mode(source: Path) -> str:
    """Classify eligible full positives for shortfall ordering and mix control."""
    try:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "real"
    if config.get("occlusion_style") == "inner_ring_center":
        return "center_occlusion"
    if config.get("marker_style") not in {"cctag_source", "classic", "reference"}:
        return "real"
    path = str(source).lower()
    if "degraded" in path or config.get("degradation_preset") == "soft_focus":
        return "degraded"
    return "clean"


@lru_cache(maxsize=None)
def is_corner_boundary_source(source: Path) -> bool:
    try:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(config.get("force_bottom_corner_target"))


@lru_cache(maxsize=None)
def is_inner_ring_occlusion_source(source: Path) -> bool:
    try:
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("occlusion_style") == "inner_ring_center"


def record_mode(record: SourceRow) -> str:
    if is_inner_ring_occlusion_source(record.source):
        visible = _float(record.row, "visible_marker_ratio", 1.0)
        clamped = int(record.row.get("target_clamped") or 0)
        occ = _float(record.row, "occlusion_ratio", 0.0)
        if occ > 0.0 and visible >= FULL_VISIBLE and clamped == 0:
            return "inner_occlusion"
    min_visible = 0.20 if is_corner_boundary_source(record.source) else MIN_VISIBLE
    mode = row_mode(record.row, min_visible=min_visible)
    if mode == "boundary" and is_corner_boundary_source(record.source):
        x, y = float(record.row["x"]), float(record.row["y"])
        if y != 639.0 or x not in (0.0, 1023.0):
            return "rejected_invalid_target"
    return mode


def load_exclusions(list_paths: list[Path]) -> set[tuple[Path, str]]:
    excluded: set[tuple[Path, str]] = set()
    for list_path in list_paths:
        with list_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    dataset_dir, filename = line.split("\t", maxsplit=1)
                    excluded.add((Path(dataset_dir).resolve(), filename))
    return excluded


def load_sources(source_dirs: list[Path], excluded: set[tuple[Path, str]] | None = None) -> tuple[list[SourceRow], list[str]]:
    excluded = excluded or set()
    records: list[SourceRow] = []
    header: list[str] | None = None
    for source in source_dirs:
        if source.name == "6f_labeled_1024x640_roi_occ" or "_roi_occ" in str(source):
            raise ValueError(f"Physical-occlusion source is forbidden: {source}")
        labels_path = source / "labels.csv"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
        with labels_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current_header = list(reader.fieldnames or [])
            if header is None:
                header = current_header
            elif current_header != header:
                raise ValueError(f"CSV header mismatch: {labels_path}")
            records.extend(SourceRow(source, row) for row in reader if (source.resolve(), row["filename"]) not in excluded)
    if not records or not header:
        raise ValueError("No label rows found")
    return records, header


def audit_records(records: list[SourceRow]) -> dict:
    report: dict[str, dict] = {}
    for record in records:
        key = str(record.source)
        item = report.setdefault(key, {"total_negatives": 0, "scale_bins": [{"total_positives": 0, "total_negatives": 0, "full_positives": 0, "inner_ring_occlusion_positives": 0, "valid_partial_boundary_positives": 0, "rejected_low_visibility_positives": 0, "rejected_occluded_positives": 0, "rejected_invalid_target_positives": 0, "target_clamped": 0} for _ in BIN_WEIGHTS]})
        index = scale_bin(record.row)
        # Negatives have no meaningful scale; report them in every-source aggregate bin 0.
        index = 0 if index is None else index
        bucket = item["scale_bins"][index]
        mode = record_mode(record)
        if mode == "negative":
            item["total_negatives"] += 1
            bucket["total_negatives"] += 1
            continue
        bucket["total_positives"] += 1
        if int(record.row.get("target_clamped") or 0):
            bucket["target_clamped"] += 1
        names = {"full": "full_positives", "inner_occlusion": "inner_ring_occlusion_positives", "boundary": "valid_partial_boundary_positives", "rejected_low_visibility": "rejected_low_visibility_positives", "rejected_occluded": "rejected_occluded_positives", "rejected_invalid_target": "rejected_invalid_target_positives"}
        bucket[names[mode]] += 1
    return report


def allocate_counts(total: int) -> list[int]:
    raw = [total * weight for weight in BIN_WEIGHTS]
    counts = [int(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _take(rng: random.Random, rows: list[SourceRow], count: int, description: str) -> list[SourceRow]:
    if len(rows) < count:
        raise ValueError(f"Insufficient {description}: have {len(rows)}, need {count}")
    return rng.sample(rows, count)


def select_rows(records: list[SourceRow], positive_target: int, negative_target: int, seed: int) -> tuple[list[SourceRow], list[int]]:
    targets = list(FIXED_POSITIVE_COUNTS) if positive_target == sum(FIXED_POSITIVE_COUNTS) else allocate_counts(positive_target)
    rng = random.Random(seed)
    selected: list[SourceRow] = []
    for index, target in enumerate(targets):
        rows = [r for r in records if scale_bin(r.row) == index]
        boundary = [r for r in rows if record_mode(r) == "boundary"]
        corner_boundary = [r for r in boundary if is_corner_boundary_source(r.source)]
        # A corner-boundary regeneration replaces, rather than mixes with, the
        # legacy generic-edge boundary distribution for that scale bin.
        if corner_boundary:
            boundary = corner_boundary
        full_real = [r for r in rows if record_mode(r) == "full" and source_mode(r.source) == "real"]
        full_clean = [r for r in rows if record_mode(r) == "full" and source_mode(r.source) == "clean"]
        full_degraded = [r for r in rows if record_mode(r) == "full" and source_mode(r.source) == "degraded"]
        full_occluded = [r for r in rows if record_mode(r) == "inner_occlusion"]
        boundary_target = round(target * BOUNDARY_FRACTION)
        chosen = _take(rng, boundary, boundary_target, f"boundary positives in {BIN_EDGES[index]:g}-{BIN_EDGES[index + 1]:g}px")
        # Exact fixed-tile mode allocation: 65% clean/domain-real, 20%
        # degraded, 10% corner boundary, 5% inner-ring center occlusion.
        occluded_target = round(target * 0.05)
        degraded_target = round(target * 0.20)
        clean_target = target - boundary_target - occluded_target - degraded_target
        chosen.extend(_take(rng, full_occluded, occluded_target, f"inner-ring occlusion positives in bin {index}"))
        real_count = min(clean_target, len(full_real))
        chosen.extend(_take(rng, full_real, real_count, f"real full positives in bin {index}"))
        chosen.extend(_take(rng, full_clean, clean_target - real_count, f"clean synthetic positives in bin {index}"))
        chosen.extend(_take(rng, full_degraded, degraded_target, f"degraded synthetic positives in bin {index}"))
        selected.extend(chosen)
    negatives = [r for r in records if row_mode(r.row) == "negative"]
    selected.extend(_take(rng, negatives, negative_target, "negatives"))
    rng.shuffle(selected)
    return selected, targets


def resolve_heatmap(source: Path, stem: str) -> Path:
    for suffix in (".npz", ".npy"):
        candidate = source / "heatmaps" / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing heatmap for {source}:{stem}")


def materialize(selected: list[SourceRow], header: list[str], output: Path, config: dict, use_hardlinks: bool = True) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    images_dir, heatmaps_dir = output / "images", output / "heatmaps"
    images_dir.mkdir(parents=True, exist_ok=True); heatmaps_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    def transfer(source: Path, destination: Path) -> None:
        if use_hardlinks:
            try: os.link(source, destination); return
            except OSError: pass
        shutil.copy2(source, destination)
    for index, record in enumerate(selected):
        old, new = record.row["filename"], f"{index:06d}"
        transfer(record.source / "images" / f"{old}.png", images_dir / f"{new}.png")
        heatmap = resolve_heatmap(record.source, old)
        transfer(heatmap, heatmaps_dir / f"{new}{heatmap.suffix}")
        row = dict(record.row); row["filename"] = new; output_rows.append(row)
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header); writer.writeheader(); writer.writerows(output_rows)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Output directory; omit with --audit-only.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--positive-target", type=int, default=15000)
    parser.add_argument("--negative-target", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-list", action="append", default=[], type=Path)
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()
    if not args.audit_only and args.output is None: parser.error("--output is required unless --audit-only")
    exclusions = load_exclusions(args.exclude_list)
    try:
        records, header = load_sources(args.source, exclusions)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    audit = audit_records(records)
    print(json.dumps(audit, indent=2))
    if args.audit_only: return
    try: selected, targets = select_rows(records, args.positive_target, args.negative_target, args.seed)
    except ValueError as exc: parser.exit(2, f"error: {exc}\n")
    final_modes = []
    for index in range(len(BIN_WEIGHTS)):
        rows = [r for r in selected if scale_bin(r.row) == index and record_mode(r) != "negative"]
        final_modes.append({
            "full_real": sum(record_mode(r) == "full" and source_mode(r.source) == "real" for r in rows),
            "clean_synthetic": sum(record_mode(r) == "full" and source_mode(r.source) == "clean" for r in rows),
            "degraded_synthetic": sum(record_mode(r) == "full" and source_mode(r.source) == "degraded" for r in rows),
            "inner_ring_occlusion": sum(record_mode(r) == "inner_occlusion" for r in rows),
            "boundary": sum(record_mode(r) == "boundary" for r in rows),
        })
    config = {"output_size": "1024x640", "scale_bin_edges": list(BIN_EDGES), "positive_bin_counts": targets, "negative_count": args.negative_target, "boundary_fraction": BOUNDARY_FRACTION, "eligibility_rules": {"min_visible_marker_ratio": MIN_VISIBLE, "full_visible_marker_ratio": FULL_VISIBLE, "boundary_requires_target_clamped": 1, "full_requires_target_clamped": 0, "physical_occlusion": "rejected"}, "final_mode_counts": final_modes, "source_paths": [str(path) for path in args.source], "excluded_lists": [str(path) for path in args.exclude_list], "excluded_sample_count": len(exclusions), "rejected_counts": audit, "seed": args.seed, "artifact_mode": "copy" if args.copy_files else "hardlink_with_copy_fallback"}
    materialize(selected, header, args.output, config, not args.copy_files)
    print(f"Built {len(selected)} samples at {args.output}")


if __name__ == "__main__": main()
