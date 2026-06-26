#!/bin/bash
#SBATCH -J exp_parameter_noise_no_held              # Job name
#SBATCH -o hpc/exploration_exps/exp_parameter_noise_no_held-%A.out  # stdout log
#SBATCH -e hpc/exploration_exps/exp_parameter_noise_no_held-%A.err  # stderr log
#SBATCH -p gpu                     # GPU partition
#SBATCH --gres=gpu:a100:1          # 1x A100 GPU
#SBATCH -c 4                       # 4 CPU cores
#SBATCH --mem=32G                  # 32 GB RAM
#SBATCH -t 02:00:00                # 2 hour wall time
#SBATCH -A sds_capstone_atashman   # SU allocation group

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

# Load conda environment
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

echo "=== Experiment: Exp Parameter Noise No Held ==="
echo "Strategy: parameter_noise"
echo "Prioritize Held: No"
echo "Circuit Breaker: $$500"
echo ""

# Run replay with exploration strategy
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 500 \
    --strategy parameter_noise \
    \
    --run-name exp_parameter_noise_no_held

echo ""
echo "=== Experiment exp_parameter_noise_no_held complete ==="
echo "Checkpoints: rl_bot/checkpoints_exp_parameter_noise_no_held/"
echo "Trade log: output/rl_trades_exp_parameter_noise_no_held.csv"
