#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12.11}"
PYTORCH_TARGET="${PYTORCH_TARGET:-cu126}"

if [[ "${PYTORCH_TARGET}" != "cu126" && "${PYTORCH_TARGET}" != "cpu" ]]; then
  echo "Unsupported PYTORCH_TARGET=${PYTORCH_TARGET}. Use 'cu126' or 'cpu'." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed." >&2
  echo "Install it from https://docs.astral.sh/uv/getting-started/installation/ and rerun." >&2
  exit 1
fi

uv python install "${PYTHON_VERSION}"
uv sync --python "${PYTHON_VERSION}" --extra "${PYTORCH_TARGET}"

echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Run commands with: uv run python src/generate_cctag_dataset.py --help"
