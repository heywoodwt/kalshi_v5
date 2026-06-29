# Phase 1 Deployment - Quick Start Guide

Complete deployment in 3 steps: Test → Deploy → Monitor

---

## Prerequisites

✅ All 8 model checkpoints downloaded (`rl_bot/mm_checkpoints/*.zip`)
✅ Kalshi account with $95.82+ balance
✅ Kalshi API credentials

---

## Step 1: Paper Trading (Required - 24 Hours Minimum)

Test everything with simulated orders before risking real capital.

### 1.1 Set Environment Variables

```bash
# Create .env file
cat > .env <<EOF
# Paper mode (no real orders)
PAPER_MODE=true

# Kalshi credentials (optional for paper mode, but good to test)
KALSHI_API_KEY=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here
KALSHI_EMAIL=your_email@example.com
KALSHI_PASSWORD=your_password
EOF
```

### 1.2 Run Paper Trading

```bash
# Load environment variables
export $(cat .env | xargs)

# Run bot in paper mode
python -m rl_bot.live_trader_v2
```

### 1.3 Expected Output

```
================================================================================
KALSHI LIVE TRADING BOT - PHASE 1
================================================================================
Mode: PAPER TRADING
Capital: $95.82
Categories: 8
Risk Limits: Daily loss $10.0, Stop loss $-25.0
================================================================================

2026-06-29 12:00:02 [INFO] Using PAPER TRADING mode
2026-06-29 12:00:03 [INFO] Loading models...
2026-06-29 12:00:04 [INFO] ✓ Loaded model: KXLOLTOTALMAPS
2026-06-29 12:00:05 [INFO] ✓ Loaded model: KXAFCCLGAME
...
2026-06-29 12:00:12 [INFO] Loaded 8 models
2026-06-29 12:00:13 [INFO] Finding active markets...
2026-06-29 12:00:14 [INFO] ✓ KXAFCCLGAME: KXAFCCLGAME-24JUN29-B10
...
2026-06-29 12:00:20 [INFO] ✓ WebSocket connected
2026-06-29 12:00:21 [INFO] Starting live trading...

# Orders will appear as:
2026-06-29 12:01:15 [INFO] ORDER: KXAFCCLGAME/KXAFCCLGAME-24JUN29-B10 buy 1 @ 48¢
2026-06-29 12:01:15 [INFO] [PAPER] Order created: paper_order_1 - buy 1 KXAFCCLGAME-24JUN29-B10 @ 48
```

### 1.4 Monitor Paper Trading (24 Hours)

Let it run for 24 hours minimum. Check these:

- [ ] No crashes or errors
- [ ] WebSocket stays connected
- [ ] Orders generate based on orderbook
- [ ] Simulated fills process correctly
- [ ] PnL tracking works
- [ ] Hourly reports appear

**If any issues:** See `rl_bot/TESTING_GUIDE.md` for troubleshooting.

---

## Step 2: API Integration Tests (Before Live)

Run these tests to verify Kalshi API works correctly.

### 2.1 Test API Connection

```bash
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

# Check balance
balance = client.get_balance()
print(f'✓ Balance: \${balance[\"balance\"] / 100:.2f}')

# Check markets
markets = client.get_markets(limit=5)
print(f'✓ Markets: {len(markets[\"markets\"])} open markets found')

print('✓ API connection successful!')
"
```

**Expected:**
```
✓ Balance: $95.82
✓ Markets: 5 open markets found
✓ API connection successful!
```

### 2.2 Test Order Placement (and Cancel)

```bash
# Run the test order script
bash << 'EOF'
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os
import time

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

# Find a market
markets = client.get_markets(series_ticker='KXAFCCLGAME', status='open', limit=1)
ticker = markets['markets'][0]['ticker']
print(f'Testing on: {ticker}')

# Get orderbook
orderbook = client.get_orderbook(ticker)
best_bid = orderbook['orderbook']['yes'][0][0]

# Place order well below market (won't fill)
test_price = max(1, best_bid - 10)
print(f'Placing test order at {test_price}¢ (unlikely to fill)...')

order = client.place_limit_order(ticker=ticker, side='buy', price_cents=test_price, size=1)
order_id = order['order_id']
print(f'✓ Order placed: {order_id}')

# Wait 5 seconds
time.sleep(5)

# Check status
status = client.get_order(order_id)
print(f'✓ Status: {status[\"order\"][\"status\"]}')

# Cancel
client.cancel_order(order_id)
print(f'✓ Order canceled')

print('')
print('✓ Order placement test passed!')
"
EOF
```

**Expected:**
```
Testing on: KXAFCCLGAME-24JUN29-B10
Placing test order at 38¢ (unlikely to fill)...
✓ Order placed: order_abc123
✓ Status: resting
✓ Order canceled

✓ Order placement test passed!
```

### 2.3 Full Test Suite

```bash
# Run complete test suite
python rl_bot/kalshi_api.py
```

This runs all built-in tests in paper mode.

**If all tests pass:** Ready for live deployment!

---

## Step 3: Live Deployment

⚠️ **Only proceed if:**
- ✅ Paper trading ran successfully for 24+ hours
- ✅ API integration tests passed
- ✅ Test order placed and canceled successfully
- ✅ You understand the risks

### 3.1 Switch to Live Mode

```bash
# Update .env file
cat > .env <<EOF
# LIVE MODE - Real orders!
PAPER_MODE=false

KALSHI_API_KEY=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here
KALSHI_EMAIL=your_email@example.com
KALSHI_PASSWORD=your_password
EOF
```

### 3.2 Start with ONE Category (Recommended)

Edit `rl_bot/live_config_phase1.py`:

```python
# Comment out 7 categories, keep only the safest one:
PHASE1_CATEGORIES = [
    CategoryConfig(
        name="KXAFCCLGAME",  # Reverse overfit, proven
        test_pnl_per_episode=3.46,
        overfit_ratio=-0.60,
        test_win_rate=0.67,
        max_contracts=1,
        max_inventory=5,
        capital_allocation=15.00,
    ),
    # Comment out the rest for initial testing
]
```

### 3.3 Launch Live Trading

```bash
# Load environment
export $(cat .env | xargs)

# Run in background with logging
nohup python -m rl_bot.live_trader_v2 > trader.log 2>&1 &

# Get process ID
echo $! > trader.pid

echo "Bot running with PID: $(cat trader.pid)"
echo "Monitor with: tail -f live_trading.log"
```

### 3.4 Monitor First Hour (Critical!)

Watch very closely for the first hour:

```bash
# Watch logs in real-time
tail -f live_trading.log

# Check every 5 minutes:
# - Quotes being sent?
# - Any fills?
# - Positions tracking?
# - PnL updating?
# - No errors?
```

Expected activity:
```
2026-06-29 14:00:01 [INFO] Starting live trading...
2026-06-29 14:00:30 [INFO] ORDER: KXAFCCLGAME/KXAFCCLGAME-24JUN29-B10 buy 1 @ 48¢
2026-06-29 14:00:30 [INFO] ✓ Order placed: order_xyz789
2026-06-29 14:02:15 [INFO] FILL: KXAFCCLGAME/KXAFCCLGAME-24JUN29-B10 buy 1 @ 48¢
2026-06-29 14:02:15 [INFO] Cumulative PnL: $0.00
...
2026-06-29 15:00:00 [INFO] ================================================================================
2026-06-29 15:00:00 [INFO] HOURLY SUMMARY
2026-06-29 15:00:00 [INFO] ================================================================================
2026-06-29 15:00:00 [INFO] Quotes sent: 23
2026-06-29 15:00:00 [INFO] Fills: 3 (Fill rate: 13.0%)
2026-06-29 15:00:00 [INFO] Wins: 2, Losses: 1 (Win rate: 66.7%)
2026-06-29 15:00:00 [INFO] Daily PnL: $1.20
```

### 3.5 Emergency Stop (If Needed)

```bash
# Stop the bot immediately
kill $(cat trader.pid)

# Cancel all open orders
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os
client = KalshiRESTClient(os.getenv('KALSHI_API_KEY'), os.getenv('KALSHI_API_SECRET'))
client.cancel_all_orders()
print('All orders canceled')
"
```

### 3.6 Expand to All 8 Categories

**After 24 hours of successful single-category trading:**

1. Restore full `PHASE1_CATEGORIES` list in `live_config_phase1.py`
2. Restart the bot
3. Monitor for first hour again
4. Check that all 8 categories are active

---

## Monitoring & Maintenance

### Daily Checks

```bash
# Check if bot is still running
ps aux | grep live_trader_v2

# View last 100 log lines
tail -100 live_trading.log

# Check today's performance
grep "Daily PnL" live_trading.log | tail -1
```

### Weekly Review

```bash
# Calculate weekly PnL
grep "FILL:" live_trading.log | wc -l  # Total fills

# Check fill rate
python -c "
import re
with open('live_trading.log') as f:
    logs = f.read()
    fills = len(re.findall(r'FILL:', logs))
    quotes = len(re.findall(r'ORDER:', logs))
    print(f'Fill rate: {fills}/{quotes} = {fills/quotes*100:.1f}%')
"

# Review for issues
grep "ERROR" live_trading.log | tail -20
grep "WARNING" live_trading.log | tail -20
```

### Restart Bot (After Changes)

```bash
# Stop current instance
kill $(cat trader.pid)

# Restart
nohup python -m rl_bot.live_trader_v2 > trader.log 2>&1 &
echo $! > trader.pid
```

---

## Performance Expectations

### Conservative (15% Fill Rate)

| Metric | Expected Value |
|--------|----------------|
| Daily PnL | $12.00 |
| Monthly PnL | $360 |
| Annual PnL | $4,381 |
| ROI | 4,572% |
| Fill rate | 10-20% |
| Win rate | 50-70% |

### Realistic First Week

- Daily PnL: $5-15 (may vary widely)
- Fill rate: 5-15% (lower than backtest)
- Win rate: 40-60% (may be lower initially)
- Some losing days expected

**If after 1 week:**
- Cumulative PnL negative: Investigate, may need to adjust
- Fill rate < 5%: Liquidity issue, check markets
- Win rate < 40%: Strategy not working, halt and investigate

---

## Troubleshooting

### Bot won't start

```bash
# Check logs for error
cat trader.log

# Common issues:
# - Missing .env file -> Create it
# - Invalid API credentials -> Check .env
# - Missing checkpoints -> Run download script
```

### No fills

```bash
# Check if orders are being placed
grep "ORDER:" live_trading.log | tail -10

# Check if markets are open
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os
client = KalshiRESTClient(os.getenv('KALSHI_API_KEY'), os.getenv('KALSHI_API_SECRET'))
markets = client.get_markets(series_ticker='KXAFCCLGAME', status='open')
print(f'Open markets: {len(markets[\"markets\"])}')
"
```

### WebSocket disconnects

```bash
# Bot should auto-reconnect
# If not, restart bot
kill $(cat trader.pid)
nohup python -m rl_bot.live_trader_v2 > trader.log 2>&1 &
echo $! > trader.pid
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `rl_bot/live_trader_v2.py` | Main trading bot |
| `rl_bot/kalshi_api.py` | Kalshi REST API client |
| `rl_bot/live_config_phase1.py` | Configuration (8 categories, capital allocation) |
| `rl_bot/TESTING_GUIDE.md` | Comprehensive testing procedures |
| `rl_bot/DEPLOY_PHASE1.md` | Detailed deployment guide |
| `live_trading.log` | Bot runtime logs (created when running) |
| `.env` | API credentials (create this) |

---

## Support

**For issues:**
1. Check logs: `tail -100 live_trading.log`
2. See troubleshooting in `TESTING_GUIDE.md`
3. Emergency stop: `kill $(cat trader.pid)`

**Good luck! 🚀**
