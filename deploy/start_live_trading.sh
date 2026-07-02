#!/bin/bash
# Start Kalshi Live Trading Bot
# ⚠️  LIVE MODE - REAL MONEY TRADING

set -e

echo "=========================================="
echo "Starting Kalshi MM Trading Bot"
echo "Mode: LIVE (REAL MONEY)"
echo "Categories: 8"
echo "=========================================="

# Activate virtual environment
source ~/venv/bin/activate 2>/dev/null || source venv/bin/activate

# Check environment
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found"
    exit 1
fi

# Verify live mode
if grep -q "PAPER_MODE=true" .env; then
    echo "ERROR: PAPER_MODE is still true"
    echo "Set PAPER_MODE=false in .env for live trading"
    exit 1
fi

# Verify credentials
if ! grep -q "KALSHI_API_KEY=" .env || ! grep -q "KALSHI_API_SECRET=" .env; then
    echo "ERROR: Missing Kalshi API credentials in .env"
    exit 1
fi

echo
echo "Environment verified:"
echo "  ✓ Live mode enabled"
echo "  ✓ API credentials present"
echo

# Start trading bot
echo "Starting live trading bot..."
echo "Logs: live_trading.log"
echo

python -m rl_bot.live_trader_v2

echo
echo "Trading bot stopped"
