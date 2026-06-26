#!/bin/bash
#SBATCH -J exp_nofees            # Job name
#SBATCH -o exp_nofees-%A.out     # stdout log
#SBATCH -e exp_nofees-%A.err     # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -A sds_capstone_atashman
#
# No fees test: maker/taker fees = 0, spread = 0.
# Isolates whether the model has directional signal
# independent of transaction costs.

set -euo pipefail
cd /scratch/mtk9va/kalshi_v5
module purge && module load miniforge
eval "$(conda shell.bash hook)" && conda activate kalshi_rl

echo "=== No Fees Test ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 999999 \
    --prioritize-held \
    --no-fees \
    --run-name exp_nofees
