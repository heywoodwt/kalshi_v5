#!/bin/bash
#SBATCH -J exp2_held             # Job name
#SBATCH -o exp2_held-%A.out      # stdout log (%A = job ID)
#SBATCH -e exp2_held-%A.err      # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1        # 1x A100 GPU
#SBATCH -c 4                     # 4 CPU cores
#SBATCH --mem=32G                # 32 GB RAM
#SBATCH -t 02:00:00              # 2 hour wall time
#SBATCH -A sds_capstone_atashman # SU allocation group
#
# Experiment 2: Keep max_markets_per_step=5, but prioritize markets
# where the agent holds an open position.  Remaining slots filled by vol.
# Hypothesis: agent will get opportunity to close positions and learn from
# realized PnL even with the 5-market cap.

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Experiment 2: Prioritize Held Markets ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 999999 \
    --prioritize-held \
    --run-name exp2_prioritize_held

echo "=== Experiment 2 complete ==="
