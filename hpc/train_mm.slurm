#!/bin/bash
#SBATCH -J mm_train_all
#SBATCH -o hpc/mm_results/mm_train_all-%A.out
#SBATCH -e hpc/mm_results/mm_train_all-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -A sds_capstone_atashman

# Train MM PPO agent across all market categories with 2 PPO epochs.
# Runs sequentially through each discovered category prefix.
# Uses 1x A100 GPU, 4 CPU cores, 32GB RAM, 12h wall time.
#
# Usage:
#   1. bash hpc/deploy.sh                    # deploy to HPC
#   2. ssh mtk9va@login.hpc.virginia.edu
#   3. cd /scratch/mtk9va/kalshi_v5
#   4. mkdir -p hpc/mm_results
#   5. sbatch hpc/train_mm.slurm

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

# Create results dir if needed
mkdir -p hpc/mm_results

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

# Show GPU info
echo "=== GPU Info ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
"

# Train MM PPO agent for all categories
echo ""
echo "=== Starting MM training (all categories, 2 PPO epochs) ==="
python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_3mo.parquet \
    --category all \
    --timesteps 500000 \
    --ppo-epochs 2 \
    --run-name mm_2epoch

echo ""
echo "=== Training complete ==="
echo "Checkpoints: $PROJECT_DIR/rl_bot/mm_checkpoints/"
ls -lh rl_bot/mm_checkpoints/*.zip 2>/dev/null || echo "No checkpoints found"
