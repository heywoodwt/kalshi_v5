#!/bin/bash
#SBATCH -J exp_feeaware          # Job name
#SBATCH -o exp_feeaware-%A.out   # stdout log
#SBATCH -e exp_feeaware-%A.err   # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00              # 6 hours (10 epochs * ~24 min/epoch + margin)
#SBATCH -A sds_capstone_atashman
#
# Fee-aware reward + 10-epoch training.
# Front-loads roundtrip cost at open time so the agent learns
# that trading has a cost immediately (not delayed until close).
# Multi-epoch gives the model 10x more training experience.

set -euo pipefail
cd /scratch/mtk9va/kalshi_v5
module purge && module load miniforge
eval "$(conda shell.bash hook)" && conda activate kalshi_rl

echo "=== Fee-Aware + 10-Epoch Training ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 999999 \
    --prioritize-held \
    --epochs 10 \
    --run-name exp_fee_aware
