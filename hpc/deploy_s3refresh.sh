#!/bin/bash
# Deploy the S3-refresh retraining job (code + the two refreshed parquets).
#
# Usage:  bash hpc/deploy_s3refresh.sh [--submit]

set -euo pipefail

HPC_USER="mtk9va"
HPC_HOST="login.hpc.virginia.edu"
REMOTE_DIR="/scratch/$HPC_USER/kalshi_v5"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying S3-refresh job to UVA HPC ==="

# 1. Sync code (exclude output/ and other heavy or irrelevant dirs)
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

# 2. Push the refreshed data (overwrites the 3-day versions on HPC)
rsync -avz --progress \
    "$LOCAL_DIR/output/rl_kalshi_trades_s3.parquet" \
    "$LOCAL_DIR/output/s3_orderbooks.parquet" \
    "$HPC_USER@$HPC_HOST:$REMOTE_DIR/output/"

if [ "${1:-}" = "--submit" ]; then
    echo "=== Submitting SLURM array job ==="
    ssh "$HPC_USER@$HPC_HOST" "cd $REMOTE_DIR && sbatch hpc/train_mm_s3refresh.slurm"
fi

echo "=== Done ==="
echo "Pull results when finished:"
echo "  rsync -avz $HPC_USER@$HPC_HOST:$REMOTE_DIR/rl_bot/mm_checkpoints/realistic_20dim_*_final.zip rl_bot/mm_checkpoints/"
echo "  rsync -avz '$HPC_USER@$HPC_HOST:$REMOTE_DIR/output/realistic_20dim_*_eval.csv' output/"
