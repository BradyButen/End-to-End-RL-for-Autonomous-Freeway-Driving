#!/bin/bash
# One-time environment setup on Great Lakes (run on a LOGIN node: it needs internet for pip + MetaDrive assets).
#
#   bash slurm/setup_greatlakes.sh            # CUDA build of PyTorch (for the gpu / spgpu partitions)
#   CPU_ONLY=1 bash slurm/setup_greatlakes.sh # CPU-only PyTorch (standard partition)
#
# Check before running:
#   * `module avail python` — pick a 3.10–3.12 module (MetaDrive is pinned to a GitHub commit that supports 3.12).
#   * the CUDA wheel index matching the cluster driver (`nvidia-smi` on a GPU node); cu121 is a safe default.
set -euo pipefail

PY_MODULE="${PY_MODULE:-python/3.12}"     # TODO: confirm the exact module name with `module avail python`
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$REPO_DIR/.venv}"

module load "$PY_MODULE"
python3 --version

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

if [[ "${CPU_ONLY:-0}" == "1" ]]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch --index-url "$CUDA_INDEX"
fi
pip install -e "$REPO_DIR[dev]"

# MetaDrive downloads its model/texture assets on first use; compute nodes may have no internet, so do it here.
python -m metadrive.pull_asset

# Smoke: imports + unit tests (no simulator) + one headless MetaDrive episode
SDL_VIDEODRIVER=dummy python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pytest -q
SDL_VIDEODRIVER=dummy python scripts/validate.py --check smoke
echo "Setup done: source $VENV/bin/activate"
