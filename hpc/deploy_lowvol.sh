#!/bin/bash
# Deploy the low-vol 20-dim retraining job (Phase 3) to UVA HPC.
# Code-only rsync: both trades parquets and the S3 orderbooks are already on
# HPC (3mo from earlier mm_* jobs, S3 files from the KXBTCD deploy).
#
# Usage:  bash hpc/deploy_lowvol.sh [--submit]
#   --submit also runs sbatch remotely after syncing.

set -euo pipefail

HPC_USER="mtk9va"
HPC_HOST="login.hpc.virginia.edu"
REMOTE_DIR="/scratch/$HPC_USER/kalshi_v5"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying low-vol retraining job to UVA HPC ==="

# Sync code (exclude output/ and other heavy or irrelevant dirs)
rsync -avz \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'output' \
    --exclude 'rsa_keys' \
    --exclude 'lightning_logs' \
    --exclude 'docs' \
    --exclude '.idea' \
    --exclude '*.log' \
    --exclude 'rl_bot/checkpoints_old_unrealistic' \
    --exclude 'rl_bot/mm_checkpoints' \
    "$LOCAL_DIR/" \
    "$HPC_USER@$HPC_HOST:$REMOTE_DIR/"

if [ "${1:-}" = "--submit" ]; then
    echo "=== Submitting SLURM array job ==="
    ssh "$HPC_USER@$HPC_HOST" "cd $REMOTE_DIR && sbatch hpc/train_mm_lowvol.slurm"
fi

echo "=== Done ==="
echo "After the jobs finish, pull the models back with:"
echo "  rsync -avz $HPC_USER@$HPC_HOST:$REMOTE_DIR/rl_bot/mm_checkpoints/realistic_20dim_*_final.zip rl_bot/mm_checkpoints/"
