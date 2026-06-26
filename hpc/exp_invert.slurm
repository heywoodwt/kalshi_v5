#!/bin/bash
#SBATCH -J exp_invert            # Job name
#SBATCH -o exp_invert-%A.out     # stdout log
#SBATCH -e exp_invert-%A.err     # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -A sds_capstone_atashman
#
# Inverted policy test: pick argmin(Q) instead of argmax(Q).
# If PnL goes positive, the model learned real but backwards signal.
# If PnL stays negative, losses are purely transaction costs.

set -euo pipefail
cd /scratch/mtk9va/kalshi_v5
module purge && module load miniforge
eval "$(conda shell.bash hook)" && conda activate kalshi_rl

echo "=== Inverted Policy Test ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 999999 \
    --prioritize-held \
    --invert-policy \
    --run-name exp_invert
