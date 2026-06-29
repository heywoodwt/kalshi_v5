#!/bin/bash
# Download Phase 1 model checkpoints from HPC
# Run this script from project root: bash scripts/download_checkpoints.sh

set -e

HPC_USER="mtk9va"
HPC_HOST="login.hpc.virginia.edu"
HPC_PATH="/scratch/mtk9va/kalshi_v5/rl_bot/mm_checkpoints"
LOCAL_PATH="rl_bot/mm_checkpoints"

# Phase 1 categories (8 total)
CATEGORIES=(
    "KXLOLTOTALMAPS"
    "KXBBCHARTPOSITIONALBUM"
    "KXRAINLAXM"
    "KXNBAMVP"
    "KXSPOTIFYW"
    "KXAFCCLGAME"
    "KXATPMATCH"
    "KXATPCHALLENGERMATCH"
)

echo "================================================"
echo "Downloading Phase 1 Model Checkpoints from HPC"
echo "================================================"
echo "HPC: $HPC_USER@$HPC_HOST:$HPC_PATH"
echo "Local: $LOCAL_PATH"
echo "Categories: ${#CATEGORIES[@]}"
echo "================================================"
echo

# Create local directory
mkdir -p "$LOCAL_PATH"

# Download each checkpoint
for cat in "${CATEGORIES[@]}"; do
    checkpoint="${cat}.zip"
    remote="${HPC_USER}@${HPC_HOST}:${HPC_PATH}/${checkpoint}"
    local="${LOCAL_PATH}/${checkpoint}"

    if [ -f "$local" ]; then
        echo "✓ $checkpoint (already exists)"
    else
        echo "⬇ Downloading $checkpoint..."
        if rsync -avz --progress "$remote" "$local"; then
            echo "✓ $checkpoint downloaded"
        else
            echo "✗ Failed to download $checkpoint"
            exit 1
        fi
    fi
done

echo
echo "================================================"
echo "Download Summary"
echo "================================================"
ls -lh "$LOCAL_PATH"/*.zip
echo
echo "Total checkpoints: $(ls -1 "$LOCAL_PATH"/*.zip | wc -l | tr -d ' ')"
echo "Required: ${#CATEGORIES[@]}"
echo

# Verify all required checkpoints
missing=0
for cat in "${CATEGORIES[@]}"; do
    if [ ! -f "$LOCAL_PATH/${cat}.zip" ]; then
        echo "✗ Missing: ${cat}.zip"
        missing=$((missing + 1))
    fi
done

if [ $missing -eq 0 ]; then
    echo "✓ All Phase 1 checkpoints downloaded successfully!"
    echo
    echo "Next steps:"
    echo "1. Review configuration: python rl_bot/live_config_phase1.py"
    echo "2. Set up .env file with Kalshi API credentials"
    echo "3. Read deployment guide: rl_bot/DEPLOY_PHASE1.md"
else
    echo "✗ Missing $missing checkpoints"
    exit 1
fi
