#!/bin/bash
#SBATCH -J mm_lowvol_20dim
#SBATCH -o hpc/mm_results/mm_lowvol_20dim-%A_%a.out
#SBATCH -e hpc/mm_results/mm_lowvol_20dim-%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -A sds_capstone_atashman
#SBATCH --array=0-3

# Train 20-dim bias-corrected PPO models for the remaining Phase 2 whitelist
# categories (KXBTCD already trained as realistic_20dim_KXBTCD_final.zip).
# Sim includes ALL Phase 3 fixes:
#   - through-fills execute at OUR quote price (not the print price)
#   - fees at live order granularity (quote_size=1 -> per-contract ceil)
#   - episode-end flatten pays the TAKER rate (it crosses the spread)
#   - taker_fill_prob 0.10 (post-only quoting: only risk exits cross)
#   - post-fill inventory marking (adverse selection charged)
#
# Data per category:
#   KXWCGAME              -> 3-day S3 parquet (World Cup live: 47k trades),
#                            temporal split at 2026-07-01 (same as KXBTCD)
#   KXADP/KXAAAGASM/GASD  -> 3-month parquet (low-frequency markets: only
#                            1k-9k trades even over 3 months), split 2026-06-15
#
# Usage:
#   1. bash hpc/deploy_lowvol.sh              # from local mac
#   2. ssh mtk9va@login.hpc.virginia.edu
#   3. cd /scratch/mtk9va/kalshi_v5 && sbatch hpc/train_mm_lowvol.slurm

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

mkdir -p hpc/mm_results

# Per-array-task category and data selection
CATEGORIES=(KXWCGAME KXADP KXAAAGASM KXAAAGASD)
CATEGORY="${CATEGORIES[$SLURM_ARRAY_TASK_ID]}"

if [ "$CATEGORY" = "KXWCGAME" ]; then
    DATA="output/rl_kalshi_trades_s3.parquet"
    SPLIT="2026-07-01"
else
    DATA="output/rl_kalshi_trades_3mo.parquet"
    SPLIT="2026-06-15"
fi

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Training $CATEGORY (data=$DATA, split=$SPLIT) ==="
python -m rl_bot.mm_train \
    --data "$DATA" \
    --markets output/rl_all_markets_3mo.parquet \
    --category "$CATEGORY" \
    --timesteps 500000 \
    --ppo-epochs 10 \
    --split-date "$SPLIT" \
    --split-mode train \
    --run-name realistic_20dim \
    --orderbooks output/s3_orderbooks.parquet

# Eval on the held-out test split
echo ""
echo "=== Evaluating $CATEGORY on test split ==="
python -m rl_bot.mm_eval \
    --model "rl_bot/mm_checkpoints/realistic_20dim_${CATEGORY}_final.zip" \
    --data "$DATA" \
    --markets output/rl_all_markets_3mo.parquet \
    --category "$CATEGORY" \
    --split-date "$SPLIT" \
    --orderbooks output/s3_orderbooks.parquet \
    --output "output/realistic_20dim_${CATEGORY,,}_eval.csv"

echo ""
echo "=== $CATEGORY complete ==="
ls -lh "rl_bot/mm_checkpoints/realistic_20dim_${CATEGORY}"*.zip 2>/dev/null || echo "No checkpoints found"
