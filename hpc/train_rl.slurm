#!/bin/bash
#SBATCH -J rl_train              # Job name
#SBATCH -o rl_train-%A.out       # stdout log (%A = job ID)
#SBATCH -e rl_train-%A.err       # stderr log
#SBATCH -p gpu                   # GPU partition
#SBATCH --gres=gpu:a100:1        # 1x A100 GPU
#SBATCH -c 4                     # 4 CPU cores
#SBATCH --mem=32G                # 32 GB RAM
#SBATCH -t 01:00:00              # 1 hour wall time (job finishes in ~2 min)
#
# Full DGX node (uncomment when scaling up network/batch size):
##SBATCH --gres=gpu:a100:8       # 8x A100 80GB GPUs (full DGX node)
##SBATCH -C gpupod               # Target DGX BasePOD nodes (A100 80GB + 2TB RAM)
##SBATCH -c 64                   # 64 CPU cores
##SBATCH --mem=512G              # 512 GB RAM
##SBATCH -t 12:00:00             # 12 hour wall time
#SBATCH -A sds_capstone_atashman  # SU allocation group
#
# Train DQN agent on historical Kalshi BTC trades.
# Uses full DGX A100 node: 8 GPUs, 64 cores, 512 GB RAM.
#
# Submit:   sbatch hpc/train_rl.slurm
# Monitor:  squeue -u mtk9va
# Cancel:   scancel <job_id>

set -euo pipefail

# Project lives on scratch for fast I/O
PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

# Show all visible GPUs
echo "=== GPU Info ==="
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, '
          f'{torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB')
"

# Replay training on historical data
# --data   : path to saved parquet (10.4M trades, 323 MB)
# --speed 0: no delay between steps (train as fast as possible)
echo ""
echo "=== Starting RL replay training ==="
python -m rl_bot.replay \
    --data output/rl_kalshi_trades_3mo.parquet \
    --speed 0 \
    --max-loss 200

echo ""
echo "=== Training complete ==="
echo "Checkpoints: $PROJECT_DIR/rl_bot/checkpoints/"
echo "Trade log:   $PROJECT_DIR/output/rl_replay_trades.csv"
