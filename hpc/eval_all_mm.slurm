#!/bin/bash
#SBATCH -J eval_all_mm
#SBATCH -o hpc/mm_results/eval_all_mm-%A.out
#SBATCH -e hpc/mm_results/eval_all_mm-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -A sds_capstone_atashman

# Evaluate all trained MM models on the test split (last ~2 hours)

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

SPLIT_DATE="2026-06-30T01:24:17"
DATA="output/collector_trades.parquet"
MARKETS="output/rl_all_markets_3mo.parquet"
RESULTS_DIR="output/mm_eval_results"
mkdir -p "$RESULTS_DIR"

# Header for combined CSV
echo "category,ticker,steps,total_reward,realized_pnl,unrealized_pnl,total_pnl,final_inventory,fills_buy,fills_sell" > "$RESULTS_DIR/all_eval.csv"

# Only evaluate the 130 categories from the current training run
# Extract category list from submit script (each line: sbatch hpc/mm_jobs/mm_CAT.slurm)
CATEGORIES=$(grep '^sbatch' hpc/submit_all_mm.sh | sed 's|.*mm_jobs/mm_||; s|\.slurm||')

for cat in $CATEGORIES; do
    model_path="rl_bot/mm_checkpoints/mm_${cat}_${cat}_final.zip"
    if [ ! -f "$model_path" ]; then
        echo "SKIP: $cat (no checkpoint)"
        continue
    fi

    echo "=== Evaluating $cat ==="

    python -m rl_bot.mm_eval \
        --model "$model_path" \
        --data "$DATA" \
        --markets "$MARKETS" \
        --category "$cat" \
        --split-date "$SPLIT_DATE" \
        --output "$RESULTS_DIR/eval_${cat}.csv" \
        2>&1 || echo "FAILED: $cat"

    # Append to combined CSV (skip header)
    if [ -f "$RESULTS_DIR/eval_${cat}.csv" ]; then
        tail -n +2 "$RESULTS_DIR/eval_${cat}.csv" >> "$RESULTS_DIR/all_eval.csv"
    fi
done

echo ""
echo "=== All evaluations complete ==="

# Summary
python3 -c "
import polars as pl
df = pl.read_csv('$RESULTS_DIR/all_eval.csv')
print(f'Total episodes: {len(df)}')
print(f'Total PnL: {df[\"total_pnl\"].sum():+.4f}')
print(f'Mean reward: {df[\"total_reward\"].mean():+.4f}')
win = (df['total_pnl'] > 0).sum()
print(f'Win rate: {win}/{len(df)} = {win/max(len(df),1):.1%}')
print()

# Per-category summary
cats = df.with_columns(
    pl.col('ticker').str.split('-').list.first().alias('cat_prefix')
)
# Group by the model category (use first part of ticker before first hyphen as proxy)
# Actually just group by output file
print('Top 20 by PnL:')
by_cat = df.group_by(pl.col('ticker').str.extract(r'^([A-Z0-9]+)-')).agg(
    pl.col('total_pnl').sum().alias('pnl'),
    pl.col('total_reward').mean().alias('avg_reward'),
    pl.len().alias('episodes'),
).sort('pnl', descending=True)
print(by_cat.head(20))
print()
print('Bottom 10 by PnL:')
print(by_cat.tail(10))
"