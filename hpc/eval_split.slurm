#!/bin/bash
#SBATCH -J eval_split
#SBATCH -o hpc/mm_results/eval_split-%A.out
#SBATCH -e hpc/mm_results/eval_split-%A.err
#SBATCH -p standard
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -A sds_capstone_atashman

# Evaluate all completed MM checkpoints on train AND test splits
# to measure overfitting (train PnL >> test PnL = red flag)

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Evaluating Train/Test Split Performance ==="
echo "Start time: $(date)"
echo ""

python3 evaluate_split.py \
    --checkpoints rl_bot/mm_checkpoints \
    --data output/rl_kalshi_trades_3mo.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --split-date 2026-04-23 \
    --output hpc/mm_results/split_eval.csv

echo ""
echo "=== Evaluation Complete ==="
echo "End time: $(date)"
