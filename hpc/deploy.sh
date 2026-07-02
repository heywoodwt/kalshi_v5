#!/bin/bash
# Transfer project files to UVA HPC.
# Run this from your LOCAL machine (macOS).
#
# Prerequisites:
#   - UVA VPN connected (if off-grounds)
#   - SSH key or password for mtk9va@login.hpc.virginia.edu
#
# Usage:  bash hpc/deploy.sh

set -euo pipefail

HPC_USER="mtk9va"
HPC_HOST="login.hpc.virginia.edu"
REMOTE_DIR="/scratch/$HPC_USER/kalshi_v5"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying to UVA HPC ==="
echo "Local:  $LOCAL_DIR"
echo "Remote: $HPC_USER@$HPC_HOST:$REMOTE_DIR"
echo ""

# Create remote directory
ssh "$HPC_USER@$HPC_HOST" "mkdir -p $REMOTE_DIR"

# Sync project code (exclude large/unnecessary files)
rsync -avz --progress \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'rsa_keys' \
    --exclude 'lightning_logs' \
    --exclude 'fft' \
    --exclude 'docs' \
    --exclude '.idea' \
    --exclude 'output/*.csv' \
    --exclude 'output/*.png' \
    --exclude 'rl_bot/checkpoints_old_unrealistic' \
    "$LOCAL_DIR/" \
    "$HPC_USER@$HPC_HOST:$REMOTE_DIR/"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps:"
echo "  1. ssh $HPC_USER@$HPC_HOST"
echo "  2. cd $REMOTE_DIR"
echo "  3. bash hpc/setup_hpc.sh          # one-time env setup"
echo "  4. Edit hpc/train_rl.slurm        # set your allocation group"
echo "  5. sbatch hpc/train_rl.slurm      # submit training job"
echo "  6. squeue -u $HPC_USER            # monitor job"
