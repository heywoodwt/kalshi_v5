#!/bin/bash
#SBATCH -J exp3_combo            # Job name
#SBATCH -o exp3_combo-%A.out     # stdout log (%A = job ID)
#SBATCH -e exp3_combo-%A.err     # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1        # 1x A100 GPU
#SBATCH -c 4                     # 4 CPU cores
#SBATCH --mem=32G                # 32 GB RAM
#SBATCH -t 02:00:00              # 2 hour wall time
#SBATCH -A sds_capstone_atashman # SU allocation group
#
# Experiment 3: No market limit + prioritize held positions.
# With no cap all active markets are acted on, so prioritization only
# affects ordering (held markets are processed first).  This tests
# whether both changes together yield different results than either alone.

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Experiment 3: No Limit + Prioritize Held ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 200 \
    --no-market-limit \
    --prioritize-held \
    --run-name exp3_combined

echo "=== Experiment 3 complete ==="
