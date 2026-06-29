#!/bin/bash
# Download Phase 1 checkpoints (handles both naming conventions)

HPC_USER="mtk9va"
HPC_HOST="login.hpc.virginia.edu"
HPC_PATH="/scratch/mtk9va/kalshi_v5/rl_bot/mm_checkpoints"
LOCAL_PATH="rl_bot/mm_checkpoints"

mkdir -p "$LOCAL_PATH"

echo "Downloading Phase 1 checkpoints..."
echo

# Categories with simple names
for cat in KXAFCCLGAME KXATPMATCH KXATPCHALLENGERMATCH KXBBCHARTPOSITIONALBUM; do
    echo "⬇ $cat..."
    rsync -avz --progress \
        "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/${cat}.zip" \
        "${LOCAL_PATH}/"
done

# Categories with mm_<CAT>_<CAT>_final.zip names
for cat in KXLOLTOTALMAPS KXNBAMVP KXRAINLAXM KXSPOTIFYW; do
    echo "⬇ $cat (from final checkpoint)..."
    rsync -avz --progress \
        "${HPC_USER}@${HPC_HOST}:${HPC_PATH}/mm_${cat}_${cat}_final.zip" \
        "${LOCAL_PATH}/${cat}.zip"
done

echo
echo "✓ Download complete!"
echo
ls -lh "$LOCAL_PATH"/*.zip
echo
echo "Total: $(ls -1 "$LOCAL_PATH"/*.zip | wc -l | tr -d ' ') checkpoints"
