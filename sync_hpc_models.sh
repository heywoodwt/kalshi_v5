#!/bin/bash
# Sync trained models from HPC to local

echo "Syncing trained models from HPC..."
echo "Source: mtk9va@login.hpc.virginia.edu:/scratch/mtk9va/kalshi_v5/rl_bot/mm_checkpoints/"
echo "Target: ./rl_bot/mm_checkpoints/"
echo ""

# Create local directory if it doesn't exist
mkdir -p rl_bot/mm_checkpoints

# Sync models from HPC
rsync -avz --progress \
    mtk9va@login.hpc.virginia.edu:/scratch/mtk9va/kalshi_v5/rl_bot/mm_checkpoints/ \
    ./rl_bot/mm_checkpoints/

echo ""
echo "Sync complete!"
echo ""
echo "Models downloaded:"
ls -lh rl_bot/mm_checkpoints/*.zip | wc -l
echo ""
echo "Model list:"
ls -1 rl_bot/mm_checkpoints/*.zip | head -20
