#!/bin/bash
# One-time MSI environment setup for PINO-heat-conduction
# Usage: bash slurm/setup_env_msi.sh   (CONDA_ENV overrides the env name)
#
# A separate env keeps this repo's top-level packages (src, data, problems)
# from colliding with another project installed under the same names.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-pino}"

echo "=== Setting up MSI environment $CONDA_ENV for $PROJECT_DIR ==="

# Load conda
module load miniforge
eval "$(conda shell.bash hook)"

# Create conda env with CUDA PyTorch
conda create -n "$CONDA_ENV" python=3.11 -y
conda activate "$CONDA_ENV"

# PyTorch with CUDA (adjust cuda version to match MSI's GPU driver)
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Remaining Python deps (not torch — already installed via conda)
pip install numpy==2.2.6 scipy==1.15.3 PyYAML==6.0.2 matplotlib==3.10.0 pandas==2.2.3 pytest==9.0.3

# Install project in editable mode (makes src/, data/ and problems/ importable)
cd "$PROJECT_DIR"
pip install -e .

# Create runs output directory
mkdir -p "$HOME/fno_runs"

echo "=== Environment ready. Activate with: conda activate $CONDA_ENV ==="
echo "=== Verify with: python -c 'import torch; print(torch.cuda.is_available())' ==="
