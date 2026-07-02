#!/bin/bash
# Start Tier 1 Live Trading Bot
# Tier 1: Top 5 performing categories from testing (~1,383 markets)

echo "="
echo "TIER 1 LIVE TRADING BOT"
echo "="
echo ""
echo "Categories: 5 top-performers"
echo "  1. KXACBGAME (20 markets) - Most reliable"
echo "  2. KXATP (846 markets) - Largest liquid, year-round"
echo "  3. KXATPCHALLENGERMATCH (483 markets) - High performance"
echo "  4. KXAFCCLGAME (9 markets) - Highest reward"
echo "  5. KXAPFDDH (25 markets) - Diversification"
echo ""
echo "Capital: \$91.66"
echo "Active: \$81.00"
echo "Reserve: \$10.66"
echo ""
echo "Expected Performance: ~\$155/day (conservative estimate)"
echo ""
echo "Starting bot..."
echo ""

python -m rl_bot.live_trader_v2
