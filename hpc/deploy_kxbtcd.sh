#!/bin/bash
# Targeted deploy for the KXBTCD 20-dim training job.
# Pushes code + only the two data files the job needs (the markets metadata
# parquet is already on HPC). Avoids the 900MB+ of stale parquets in output/.
#
# Usage:  bash hpc/deploy_kxbtcd.sh

set -euo pipefail

HPC_USER="mtk9va"
HPC_HOST="login.hpc.virginia.edu"
REMOTE_DIR="/scratch/$HPC_USER/kalshi_v5"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying KXBTCD job to UVA HPC ==="

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
    "$LOCAL_DIR/" \
    "$HPC_USER@$HPC_HOST:$REMOTE_DIR/"

# 2. Push only the two data files the job reads (markets file already remote)
rsync -avz \
    "$LOCAL_DIR/output/rl_kalshi_trades_s3.parquet" \
    "$LOCAL_DIR/output/s3_orderbooks.parquet" \
    "$HPC_USER@$HPC_HOST:$REMOTE_DIR/output/"

echo "=== Deploy complete ==="
