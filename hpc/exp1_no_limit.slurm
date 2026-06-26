#!/bin/bash
#SBATCH -J exp1_nolim            # Job name
#SBATCH -o exp1_nolim-%A.out     # stdout log (%A = job ID)
#SBATCH -e exp1_nolim-%A.err     # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1        # 1x A100 GPU
#SBATCH -c 4                     # 4 CPU cores
#SBATCH --mem=32G                # 32 GB RAM
#SBATCH -t 02:00:00              # 2 hour wall time
#SBATCH -A sds_capstone_atashman # SU allocation group
#
# Experiment 1: Remove max_markets_per_step cap.
# Agent acts on ALL active markets each step instead of top-5 by vol.
# Hypothesis: agent will revisit held markets and close positions.

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Experiment 1: No Market Limit ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 200 \
    --no-market-limit \
    --run-name exp1_no_limit

echo "=== Experiment 1 complete ==="
