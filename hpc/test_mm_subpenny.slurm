#!/bin/bash
#SBATCH -J mm_subpenny_test
#SBATCH -o hpc/mm_results/mm_test-%A.out
#SBATCH -e hpc/mm_results/mm_test-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH -A sds_capstone_atashman

# Quick test of MM subpenny version with orderbook integration.
# Runs a short training session (50k steps) on a single category to verify:
#   - MMEnv initializes correctly
#   - Market metadata loader works
#   - 16-dim observation space is built
#   - Subpenny pricing is applied
#   - PPO training loop executes
#
# Usage:
#   1. bash hpc/deploy.sh                    # deploy code to HPC
#   2. ssh mtk9va@login.hpc.virginia.edu
#   3. cd /scratch/mtk9va/kalshi_v5
#   4. mkdir -p hpc/mm_results
#   5. sbatch hpc/test_mm_subpenny.slurm
#   6. tail -f hpc/mm_results/mm_test-<job-id>.out

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

# Create results dir
mkdir -p hpc/mm_results

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

# Show environment info
echo "=== Environment Info ==="
python --version
echo "Python path: $(which python)"
echo ""

# Show GPU info
echo "=== GPU Info ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# Verify data files exist
echo "=== Checking Data Files ==="
ls -lh output/rl_kalshi_trades_3mo.parquet
ls -lh output/rl_all_markets_3mo.parquet
echo ""

# Test MM training with subpenny features
# Use KXBTC category for quick test (usually has good liquidity)
echo "=== Starting MM Subpenny Test ==="
echo "Category: KXBTC"
echo "Timesteps: 50,000 (quick test)"
echo "PPO epochs: 10"
echo "Subpenny: enabled"
echo ""

python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_3mo.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category KXBTC \
    --timesteps 50000 \
    --ppo-epochs 10 \
    --run-name mm_subpenny_test

echo ""
echo "=== Test Complete ==="
echo ""
echo "Checkpoints:"
ls -lh rl_bot/mm_checkpoints/*subpenny_test* 2>/dev/null || echo "No checkpoints found"
echo ""
echo "Logs:"
ls -lh rl_bot/mm_logs/mm_subpenny_test/ 2>/dev/null || echo "No logs found"
echo ""
echo "If test succeeded, run full training with:"
echo "  sbatch hpc/train_mm.slurm"
