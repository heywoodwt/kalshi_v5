#!/bin/bash
# Pre-flight Check - Verify everything is ready for live deployment

echo "=========================================="
echo "PRE-FLIGHT CHECK"
echo "=========================================="
echo

# Check AWS credentials
echo "[1/7] Checking AWS credentials..."
if aws sts get-caller-identity &>/dev/null; then
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    echo "  ✓ AWS configured (Account: $ACCOUNT)"
else
    echo "  ✗ AWS not configured"
    echo "  Run: aws configure"
    exit 1
fi

# Check Kalshi credentials
echo "[2/7] Checking Kalshi API credentials..."
if [ -f "mm_ppo_bot.txt" ]; then
    echo "  ✓ Private key found: mm_ppo_bot.txt"
else
    echo "  ✗ Private key not found: mm_ppo_bot.txt"
    exit 1
fi

if grep -q "KALSHI_API_KEY=3e7fb42b" .env 2>/dev/null; then
    echo "  ✓ API key configured in .env"
else
    echo "  ✗ API key not configured in .env"
    exit 1
fi

# Check model checkpoints
echo "[3/7] Checking model checkpoints..."
CHECKPOINT_COUNT=$(ls -1 rl_bot/mm_checkpoints/*.zip 2>/dev/null | wc -l | tr -d ' ')
if [ "$CHECKPOINT_COUNT" -ge 8 ]; then
    echo "  ✓ Model checkpoints found: $CHECKPOINT_COUNT"
else
    echo "  ✗ Missing checkpoints (found: $CHECKPOINT_COUNT, need: 8+)"
    exit 1
fi

# Check Python dependencies
echo "[4/7] Checking Python environment..."
if python3 -c "import stable_baselines3; import requests; import websockets" 2>/dev/null; then
    echo "  ✓ Python dependencies installed"
else
    echo "  ⚠  Some dependencies may be missing"
    echo "     (Will be installed on AWS instance)"
fi

# Check code files
echo "[5/7] Checking code files..."
REQUIRED_FILES=(
    "rl_bot/live_trader_v2.py"
    "rl_bot/kalshi_api.py"
    "rl_bot/live_config_phase1.py"
)
MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ✗ Missing: $file"
        MISSING=1
    fi
done
if [ $MISSING -eq 0 ]; then
    echo "  ✓ All required files present"
fi

# Test Kalshi API connection
echo "[6/7] Testing Kalshi API connection..."
if python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
from rl_bot.kalshi_api import KalshiRESTClient
api_key = os.getenv('KALSHI_API_KEY')
api_secret = os.getenv('KALSHI_API_SECRET')
client = KalshiRESTClient(api_key=api_key, api_secret=api_secret)
balance = client.get_balance()
print(f'  ✓ API connection successful')
print(f'  ✓ Balance: \${balance.get(\"balance\", 0) / 100:.2f}')
" 2>/dev/null; then
    :
else
    echo "  ✗ API connection failed"
    echo "     Check credentials and network connection"
    exit 1
fi

# Check deployment scripts
echo "[7/7] Checking deployment scripts..."
if [ -f "deploy/aws_deploy.sh" ] && [ -f "deploy_live.sh" ]; then
    echo "  ✓ Deployment scripts ready"
else
    echo "  ✗ Missing deployment scripts"
    exit 1
fi

echo
echo "=========================================="
echo "✅ PRE-FLIGHT CHECK PASSED"
echo "=========================================="
echo
echo "System ready for live deployment:"
echo "  • AWS Account: $ACCOUNT"
echo "  • Region: us-east-2"
echo "  • Mode: LIVE TRADING (REAL MONEY)"
echo "  • Categories: 8"
echo "  • Models: $CHECKPOINT_COUNT checkpoints"
echo
echo "To deploy:"
echo "  ./deploy_live.sh"
echo
echo "⚠️  WARNING: This will start LIVE TRADING with REAL MONEY"
echo "=========================================="
