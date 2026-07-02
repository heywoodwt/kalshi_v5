#!/bin/bash
#SBATCH -J mm_kxbtcd_20dim
#SBATCH -o hpc/mm_results/mm_kxbtcd_20dim-%A.out
#SBATCH -e hpc/mm_results/mm_kxbtcd_20dim-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -A sds_capstone_atashman

# Train MM PPO on KXBTCD (BTC daily, low-vol / MM-friendly) with the
# sim-to-real bias fixes from docs/sim_vs_live_bias.md:
#   - 20-dim obs (adds realized-volatility feature, dim 19)
#   - Reward marks inventory at the POST-fill mid (charges adverse selection)
#   - 94.5% of sim fills charged the taker fee rate (live fee drag)
#   - through-fill haircut 0.33 (realistic queue position)
# Data: fresh S3 pull (kalshi-data-prod, us-east-2), 2026-06-29..07-01.
# Split: train < 2026-07-01, test >= 2026-07-01 (80.8k / 59.4k KXBTCD trades).
# Orderbooks: 274k S3 snapshots (vs the old 831) -> real book features.
#
# Usage:
#   1. bash hpc/deploy_kxbtcd.sh                # from local mac (targeted rsync)
#   2. ssh mtk9va@login.hpc.virginia.edu
#   3. cd /scratch/mtk9va/kalshi_v5
#   4. sbatch hpc/train_mm_kxbtcd.slurm

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

# Train — 20-dim bias-corrected sim, temporal split inside the S3 window
echo ""
echo "=== Training KXBTCD 20-dim model (gamma=0.90, 500k steps) ==="
python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_s3.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category KXBTCD \
    --timesteps 500000 \
    --ppo-epochs 10 \
    --split-date 2026-07-01 \
    --split-mode train \
    --run-name realistic_20dim \
    --orderbooks output/s3_orderbooks.parquet

# Eval on the held-out test split (>= 2026-07-01)
echo ""
echo "=== Evaluating on test split ==="
python -m rl_bot.mm_eval \
    --model rl_bot/mm_checkpoints/realistic_20dim_KXBTCD_final.zip \
    --data output/rl_kalshi_trades_s3.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category KXBTCD \
    --split-date 2026-07-01 \
    --orderbooks output/s3_orderbooks.parquet \
    --output output/realistic_20dim_kxbtcd_eval.csv

echo ""
echo "=== Training complete ==="
ls -lh rl_bot/mm_checkpoints/realistic_20dim_KXBTCD*.zip 2>/dev/null || echo "No checkpoints found"
