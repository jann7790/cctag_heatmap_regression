# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic CCTag (fiducial marker) dataset generation and heatmap regression training workspace. The pipeline: generate synthetic images with marker annotations, train heatmap regression models (PyTorch), then evaluate with inference and benchmarking.

## Environment Setup

- **Python**: 3.12.11 (strict pin via `.python-version`)
- **Dependency manager**: uv + pyproject.toml
- **PyTorch**: 2.11.0 with CUDA 12.6 or CPU variants (mutually exclusive extras)

```bash
uv sync --extra cu126       # GPU
uv sync --extra cpu         # CPU-only
# or: bash scripts/bootstrap_env.sh
```

## Common Commands

```bash
# Syntax check (no test suite exists)
uv run python -m py_compile src/generate_cctag_dataset.py

# Generate a small test dataset
uv run python src/generate_cctag_dataset.py --num_images 20 --seed 42 \
  --output_dir ./outputs/datasets/demo --visualize

# Visualize labels for QC
uv run python src/visualize_random_labels.py --dataset_dir ./outputs/datasets/demo \
  --num_samples 10 --show_yolo_bbox --output ./outputs/tmp/demo_labels.jpg

# Train (DDP, multi-GPU)
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/datasets/mixed --output_dir ./outputs/runs/experiment_ddp

# Train with Focal Loss (recommended for reducing false positives)
uv run torchrun --nproc_per_node=4 src/train_cctag_heatmap_ddp.py \
  --dataset_dir ./outputs/datasets/mixed --output_dir ./outputs/runs/experiment_focal \
  --focal_loss --focal_gamma 2.0

# Inference
uv run python src/infer_cctag_heatmap.py --checkpoint ./outputs/runs/experiment/best.pt \
  --input ./path/to/images --output ./outputs/inference/result --vis --eval

# Benchmark all models against all test suites
uv run python src/benchmark.py --runs_dir ./outputs/runs --suites_dir ./outputs/testing
```

## Architecture

All source files are **standalone CLI scripts** in `src/` -- there is no shared library or package structure. Each script uses `argparse` and is invoked directly.

| Script | Purpose |
|--------|---------|
| `generate_cctag_dataset.py` | Synthetic dataset generation (images, heatmaps, YOLO labels, CSV) |
| `train_cctag_heatmap_ddp.py` | Distributed training (torchrun) with backbone selection, Focal Loss, OHEM |
| `infer_cctag_heatmap.py` | Inference with optional visualization and evaluation |
| `benchmark.py` | Batch accuracy/latency benchmarking across models and test suites |
| `visualize_random_labels.py` | Label QC grid visualization |

Shell scripts in `scripts/` orchestrate multi-step workflows (generating training sets at different difficulty levels, merging datasets, running full generate-then-train pipelines).

## Dataset Output Contract

Generated datasets contain:
- `images/` (PNG), `heatmaps/` (compressed NPZ, float16, key `heatmap`; legacy `.npy` still readable), `labels_yolo/` (TXT)
- `labels.csv` with 23 columns (coordinates, ellipse geometry, occlusion, YOLO format, negative flags, visibility)
- `config.json` with generation parameters

All generated artifacts go under `outputs/`, never the repo root.

## Coding Conventions

- 4-space indent, `snake_case` functions/variables, `UPPER_CASE` module constants
- Type hints preferred; `pathlib.Path` for filesystem operations
- CLI flags use underscores: `--output_dir`, `--num_images` (not kebab-case)
- ImageNet normalization constants used in training/inference (IMAGENET_MEAN, IMAGENET_STD)

## Validation Workflow (no formal test suite)

1. `uv run python -m py_compile src/<file>.py` -- syntax check
2. Generate small dataset with `--num_images 20 --seed 42`
3. Run `visualize_random_labels.py` to confirm label alignment
4. Check output contract (images/, heatmaps/, labels_yolo/, labels.csv, config.json)
