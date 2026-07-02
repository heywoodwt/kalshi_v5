#!/bin/bash
# Train models on June data for the 29 active categories

set -e

echo "================================================================================"
echo "TRAINING MODELS ON JUNE DATA"
echo "================================================================================"
echo ""

# Active June categories (29 total)
CATEGORIES=(
    "KXBTC"
    "KXWCADVANCE"
    "KXBTCD"
    "KXWCGAME"
    "KXMLBGAME"
    "KXMVECROSSCATEGORY"
    "KXATPMATCH"
    "KXITFWMATCH"
    "KXITFMATCH"
    "KXWCMOV"
    "KXMLBSPREAD"
    "KXWCMENTION"
    "KXMENWORLDCUP"
    "KXWCSCORE"
    "KXWCGOAL"
    "KXWCTCORNERS"
    "KXWCBTTS"
    "KXWCSPREAD"
    "KXWCROUND"
    "KXWCSOA"
    "KXWCTEAMTOTAL"
    "KXWCMOF"
    "KXWCTOTAL"
    "KXETHD"
    "KXNEXTTEAMNBA"
    "KXSOLD"
    "KXMLBHR"
    "KXMLBRFI"
    "KXMVESPORTSMULTIGAMEEXTENDED"
)

# Use June data file
export DATA_FILE="output/rl_kalshi_trades_june.parquet"
export MARKETS_FILE="output/rl_all_markets_3mo.parquet"

# Shorter training for testing (can increase later)
export TIMESTEPS=50000  # Reduced from 500k for faster iteration

echo "Training ${#CATEGORIES[@]} categories on June data"
echo "Data file: $DATA_FILE"
echo "Timesteps per category: $TIMESTEPS"
echo ""

# Train each category
for category in "${CATEGORIES[@]}"; do
    echo "================================================================================"
    echo "Training: $category"
    echo "================================================================================"

    python -m rl_bot.mm_train \
        --category "$category" \
        --total-timesteps "$TIMESTEPS" \
        --trades-file "$DATA_FILE" \
        --markets-file "$MARKETS_FILE" \
        --checkpoint-dir "rl_bot/mm_checkpoints_june"

    if [ $? -eq 0 ]; then
        echo "✓ $category training complete"
    else
        echo "✗ $category training failed"
    fi
    echo ""
done

echo "================================================================================"
echo "TRAINING COMPLETE"
echo "================================================================================"
echo ""
echo "Models saved to: rl_bot/mm_checkpoints_june/"
echo ""
echo "Next steps:"
echo "  1. Update bot config to use June checkpoints"
echo "  2. Update bot config to use only the 29 active categories"
echo "  3. Deploy bot"
