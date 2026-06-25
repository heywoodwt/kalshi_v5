#!/bin/bash
# Setup script for UVA HPC (Afton/Rivanna).
# Run this ONCE after transferring the project to /scratch/mtk9va/kalshi_v5/
#
# Usage:  bash hpc/setup_hpc.sh

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"

echo "=== UVA HPC Setup for RL Bot ==="
echo "Project dir: $PROJECT_DIR"

# Load anaconda module
module purge
module load miniforge

# Create conda environment with Python 3.11 + PyTorch
if conda info --envs | grep -q kalshi_rl; then
    echo "Conda env 'kalshi_rl' already exists, skipping creation."
else
    echo "Creating conda env 'kalshi_rl' ..."
    conda create -n kalshi_rl python=3.11 -y
fi

# Activate
source activate kalshi_rl

# Install PyTorch with CUDA support (UVA HPC has NVIDIA GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies for offline replay and MM PPO training
pip install numpy polars pyarrow httpx pandas statsmodels python-dotenv stable-baselines3 gymnasium

# Create output and checkpoint directories
mkdir -p "$PROJECT_DIR/output"
mkdir -p "$PROJECT_DIR/rl_bot/checkpoints"

echo ""
echo "=== Setup complete ==="
echo "To activate:  module load miniforge && source activate kalshi_rl"
echo "To train:     sbatch hpc/train_rl.slurm"
