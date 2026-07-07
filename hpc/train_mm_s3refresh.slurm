#!/bin/bash
#SBATCH -J mm_s3refresh
#SBATCH -o hpc/mm_results/mm_s3refresh-%A_%a.out
#SBATCH -e hpc/mm_results/mm_s3refresh-%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -A sds_capstone_atashman
#SBATCH --array=0-2

# Retrain on the refreshed S3 window (2026-06-29..07-07, 6.6M trades, 1.05M
# orderbook snapshots) with the CURRENT sim — the 7/01 KXBTCD model predates
# the Phase 3 fixes (through-fills at our quote price, fees at quote_size
# granularity, taker-rate flatten, taker_fill_prob 0.10).
#
# Categories: only those with enough fresh data to learn from.
#   KXBTCD    505k trades | KXWCGAME 181k | KXAAAGASD 2.5k
#   (KXAAAGASM 421 and KXADP 9 trades in the window — keep existing models)
# Split: train < 2026-07-06, test >= 2026-07-06 (~6 days / 2 days).
# Run name stays realistic_20dim so the live trader picks the new finals up.
#
# Usage:
#   1. bash hpc/deploy_s3refresh.sh --submit    # from local mac
#   2. pull results: rsync checkpoints + eval CSVs back (see deploy script)

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

mkdir -p hpc/mm_results

CATEGORIES=(KXBTCD KXWCGAME KXAAAGASD)
CATEGORY="${CATEGORIES[$SLURM_ARRAY_TASK_ID]}"
SPLIT="2026-07-06"

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Training $CATEGORY (S3 refresh, split=$SPLIT) ==="
python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_s3.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category "$CATEGORY" \
    --timesteps 500000 \
    --ppo-epochs 10 \
    --split-date "$SPLIT" \
    --split-mode train \
    --run-name realistic_20dim \
    --orderbooks output/s3_orderbooks.parquet

echo ""
echo "=== Evaluating $CATEGORY on test split (>= $SPLIT) ==="
python -m rl_bot.mm_eval \
    --model "rl_bot/mm_checkpoints/realistic_20dim_${CATEGORY}_final.zip" \
    --data output/rl_kalshi_trades_s3.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category "$CATEGORY" \
    --split-date "$SPLIT" \
    --orderbooks output/s3_orderbooks.parquet \
    --output "output/realistic_20dim_${CATEGORY,,}_eval.csv"

echo ""
echo "=== $CATEGORY complete ==="
ls -lh "rl_bot/mm_checkpoints/realistic_20dim_${CATEGORY}_final.zip"
