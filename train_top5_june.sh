#!/bin/bash
# Train top 5 most active June categories for testing

set -e

echo "================================================================================"
echo "TRAINING TOP 5 JUNE CATEGORIES (TEST RUN)"
echo "================================================================================"
echo ""

# Top 5 by volume
CATEGORIES=(
    "KXBTC"           # 84,839 trades
    "KXWCADVANCE"     # 68,295 trades
    "KXBTCD"          # 57,284 trades
    "KXWCGAME"        # 23,280 trades
    "KXMLBGAME"       # 21,417 trades
)

export DATA_FILE="output/rl_kalshi_trades_june.parquet"
export MARKETS_FILE="output/rl_all_markets_3mo.parquet"
export TIMESTEPS=100000

echo "Training ${#CATEGORIES[@]} categories"
echo "Timesteps: $TIMESTEPS per category"
echo ""

for category in "${CATEGORIES[@]}"; do
    echo "▶ Training: $category"
    
    python -m rl_bot.mm_train \
        --category "$category" \
        --total-timesteps "$TIMESTEPS" \
        --trades-file "$DATA_FILE" \
        --markets-file "$MARKETS_FILE" \
        --checkpoint-dir "rl_bot/mm_checkpoints_june" \
        2>&1 | grep -E "Episode|reward|Training complete|Error" || true
    
    echo ""
done

echo "✓ Training complete for top 5 categories"
