#!/bin/bash
# Quick Deploy Script - Deploy Live Trading to AWS us-east-2
# ⚠️  WARNING: THIS DEPLOYS LIVE TRADING WITH REAL MONEY

set -e

echo "=========================================="
echo "KALSHI MM TRADING BOT - LIVE DEPLOYMENT"
echo "=========================================="
echo
echo "⚠️  WARNING: You are about to deploy LIVE TRADING"
echo "   - Mode: REAL MONEY (not paper trading)"
echo "   - Categories: 8"
echo "   - Region: AWS us-east-2"
echo "   - Capital: ~$95"
echo
echo "This will start trading immediately after deployment."
echo
read -p "Type 'DEPLOY LIVE' to continue: " confirmation

if [ "$confirmation" != "DEPLOY LIVE" ]; then
    echo "Deployment cancelled."
    exit 1
fi

echo
echo "Starting deployment..."
echo

# Make scripts executable
chmod +x deploy/aws_deploy.sh
chmod +x deploy/start_live_trading.sh

# Copy live environment configuration
cp deploy/.env.live .env

# Copy private key to deployment directory
if [ ! -f "mm_ppo_bot.txt" ]; then
    echo "ERROR: mm_ppo_bot.txt (private key) not found"
    exit 1
fi

# Verify checkpoints exist
CHECKPOINT_COUNT=$(ls -1 rl_bot/mm_checkpoints/*.zip 2>/dev/null | wc -l)
if [ "$CHECKPOINT_COUNT" -lt 8 ]; then
    echo "ERROR: Missing model checkpoints (found $CHECKPOINT_COUNT, need 8+)"
    exit 1
fi

echo "✓ Environment configured for live trading"
echo "✓ Private key found"
echo "✓ Model checkpoints verified ($CHECKPOINT_COUNT)"
echo

# Run AWS deployment
./deploy/aws_deploy.sh

echo
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo
echo "Your trading bot is now LIVE on AWS us-east-2"
echo "Trading with REAL MONEY across 8 categories"
echo
echo "Monitor your deployment:"
echo "  Check the AWS console output above for:"
echo "  - Instance ID"
echo "  - Public IP address"
echo "  - SSH command"
echo
echo "To stop trading:"
echo "  1. SSH into the instance"
echo "  2. Run: pkill -f live_trader_v2.py"
echo
echo "To terminate the instance:"
echo "  Use the AWS console or the command shown above"
echo
echo "⚠️  REMEMBER: This is LIVE TRADING with REAL MONEY"
echo "   Monitor your positions regularly on kalshi.com"
echo "=========================================="
