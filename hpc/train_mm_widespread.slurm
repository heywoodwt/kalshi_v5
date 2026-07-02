#!/bin/bash
#SBATCH -J mm_realistic_btc
#SBATCH -o hpc/mm_results/mm_realistic_btc-%A.out
#SBATCH -e hpc/mm_results/mm_realistic_btc-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -A sds_capstone_atashman

# Train MM PPO on KXBTC with realistic sim-to-real fixes:
# - Fill sim: no max(1,...), subpenny count//2, 10 queue competitors
# - Exit: spread cost + maker fee on liquidation
# - Orderbook data integrated into 19-dim obs
# - gamma=0.90, temporal split before 2026-04-23
#
# Usage:
#   1. bash hpc/deploy.sh
#   2. ssh mtk9va@login.hpc.virginia.edu
#   3. cd /scratch/mtk9va/kalshi_v5
#   4. mkdir -p hpc/mm_results
#   5. sbatch hpc/train_mm_widespread.slurm

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

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

# Train with wide-spread support and temporal split
echo ""
echo "=== Training KXBTC realistic model (gamma=0.90, 500k steps) ==="
python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_3mo.parquet \
    --category KXBTC \
    --timesteps 500000 \
    --ppo-epochs 10 \
    --split-date 2026-04-23 \
    --split-mode train \
    --run-name realistic_19dim \
    --orderbooks output/collector_orderbooks.parquet

# Eval on held-out test data
echo ""
echo "=== Evaluating on test split ==="
python -m rl_bot.mm_eval \
    --model rl_bot/mm_checkpoints/realistic_19dim_KXBTC_final.zip \
    --data output/rl_kalshi_trades_3mo.parquet \
    --category KXBTC \
    --split-date 2026-04-23 \
    --output output/realistic_19dim_eval.csv

echo ""
echo "=== Training complete ==="
ls -lh rl_bot/mm_checkpoints/realistic_19dim*.zip 2>/dev/null || echo "No checkpoints found"
