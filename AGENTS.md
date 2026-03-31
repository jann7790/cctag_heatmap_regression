# Repository Guidelines

## Project Structure & Module Organization
This repository is centered on two Python entry points at the root:

- `generate_cctag_dataset.py`: generates synthetic CCTag images, heatmaps, YOLO labels, and metadata.
- `visualize_random_labels.py`: samples a generated dataset and draws center points, ellipses, and optional YOLO boxes for inspection.

Reference marker assets live under `CCTag/`, especially `CCTag/markersToPrint/generators/`. Generated artifacts should go to dataset folders such as `cctag_dataset_yolo/` or temporary work directories like `tmp/`; treat these as outputs, not source.

## Build, Test, and Development Commands
Use Python 3.11+ in a local virtual environment with OpenCV and NumPy installed.

- `python generate_cctag_dataset.py --num_images 100 --output_dir ./tmp/demo --visualize`
  Creates a small reproducible dataset and preview image.
- `python visualize_random_labels.py --dataset_dir ./tmp/demo --num_samples 10 --show_yolo_bbox`
  Produces a visual QC grid for random samples.
- `python -m py_compile generate_cctag_dataset.py visualize_random_labels.py`
  Fast syntax smoke test before submitting changes.

## Coding Style & Naming Conventions
Follow the existing Python style in this repo:

- 4-space indentation, `snake_case` for functions and variables, `UPPER_CASE` for module constants.
- Prefer small helper functions with explicit type hints, as used in both scripts.
- Keep CLI flags descriptive and kebab-free: `--output_dir`, `--num_images`, `--show_yolo_bbox`.
- Use `Path` for filesystem paths and keep output directory names explicit, for example `labels_yolo/` and `heatmaps/`.

## Testing Guidelines
There is no formal test suite yet. Validate changes with focused script runs instead of large dataset jobs.

- Run `py_compile` first.
- Generate a small dataset with a fixed seed, for example `--num_images 20 --seed 42`.
- Inspect the output contract: `images/`, `heatmaps/`, `labels_yolo/`, `labels.csv`, and `config.json`.
- Use `visualize_random_labels.py` to confirm label alignment before merging.

## Commit & Pull Request Guidelines
The available history is minimal (`init`, `Switch data encoding...`), so prefer short imperative commit subjects, for example `Add ellipse label visualization`.

For pull requests, include:

- A concise summary of behavior changes.
- Exact commands used for validation.
- Sample output paths or screenshots when image generation or visualization changes.
- Notes on dataset format changes, especially if `labels.csv`, `labels_yolo/`, or marker assets are affected.
