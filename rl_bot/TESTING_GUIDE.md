## Kalshi API Integration Testing Guide

Complete testing checklist before live deployment with real capital.

---

## Phase 0: Paper Trading Mode (Required First)

### Setup

```bash
# Set paper mode environment variable
export PAPER_MODE=true

# Optional: Set dummy credentials (paper mode doesn't need real ones)
export KALSHI_API_KEY=paper_test
export KALSHI_API_SECRET=paper_test

# Run in paper mode
python -m rl_bot.live_trader_v2
```

### What Paper Mode Does

- ✅ Simulates all API calls (no real orders)
- ✅ Tracks simulated balance ($95.82 starting capital)
- ✅ Records simulated fills and positions
- ✅ Tests all logic without risk
- ✅ Logs all would-be orders to console

### Expected Output (Paper Mode)

```
================================================================================
KALSHI LIVE TRADING BOT - PHASE 1
================================================================================
Mode: PAPER TRADING
Capital: $95.82
Categories: 8
Risk Limits: Daily loss $10.0, Stop loss $-25.0
================================================================================

2026-06-29 12:00:00 [INFO] Initializing live trader (Mode: PAPER)...
2026-06-29 12:00:01 [INFO] Using PAPER TRADING mode
2026-06-29 12:00:02 [INFO] Loading models...
2026-06-29 12:00:03 [INFO] ✓ Loaded model: KXLOLTOTALMAPS
2026-06-29 12:00:04 [INFO] ✓ Loaded model: KXAFCCLGAME
...
2026-06-29 12:00:10 [INFO] Loaded 8 models
2026-06-29 12:00:11 [INFO] Connecting to WebSocket...
2026-06-29 12:00:12 [INFO] ✓ WebSocket connected
2026-06-29 12:00:13 [INFO] ✓ Subscribed to KXAFCCLGAME-24JUN29-B10
...
2026-06-29 12:00:20 [INFO] Starting live trading...

# When orders would be placed:
2026-06-29 12:01:15 [INFO] ORDER: KXAFCCLGAME/KXAFCCLGAME-24JUN29-B10 buy 1 @ 48¢
2026-06-29 12:01:15 [INFO] [PAPER] Order created: paper_order_1 - buy 1 KXAFCCLGAME-24JUN29-B10 @ 48
2026-06-29 12:01:15 [INFO] ✓ Order placed: paper_order_1

# Simulated fills (you can manually trigger these for testing)
2026-06-29 12:02:30 [INFO] [PAPER] Fill executed: buy 1 KXAFCCLGAME-24JUN29-B10 @ 48¢
2026-06-29 12:02:30 [INFO] FILL: KXAFCCLGAME/KXAFCCLGAME-24JUN29-B10 buy 1 @ 48¢
```

### Paper Mode Tests (24 Hours)

**Run paper mode for 24 hours minimum** before considering live trading:

| Test | Pass Criteria | Status |
|------|--------------|--------|
| Models load successfully | All 8 models loaded | [ ] |
| WebSocket connects | Connection established, no disconnects | [ ] |
| Orders generate | Quotes sent based on orderbook | [ ] |
| Order tracking | Pending orders tracked correctly | [ ] |
| Fill simulation | Fills processed correctly | [ ] |
| Position tracking | Positions update on fills | [ ] |
| PnL calculation | Daily/cumulative PnL tracked | [ ] |
| Risk limits | Trading halts at -$10 daily loss | [ ] |
| No crashes | Bot runs 24h without errors | [ ] |

**If any test fails:** Fix and restart 24h paper trading from scratch.

---

## Phase 1: API Integration Testing (Before Real Orders)

### Test 1: REST API Connection

```bash
# Test basic API connection
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

# Test authentication
balance = client.get_balance()
print(f'Balance: \${balance[\"balance\"] / 100:.2f}')

# Test market data
markets = client.get_markets(limit=5)
print(f'Markets: {len(markets[\"markets\"])} found')
"
```

**Expected output:**
```
Balance: $95.82
Markets: 5 found
```

**If fails:** Check API credentials in `.env` file.

### Test 2: Get Positions

```bash
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

positions = client.get_positions()
print(f'Open positions: {len(positions[\"positions\"])}')
for pos in positions['positions']:
    print(f'  {pos[\"ticker\"]}: {pos[\"position\"]} contracts')
"
```

**Expected output:**
```
Open positions: 0
(or list of any existing positions)
```

### Test 3: Get Orderbook

```bash
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

# Get a live market ticker first
markets = client.get_markets(series_ticker='KXAFCCLGAME', status='open', limit=1)
if markets['markets']:
    ticker = markets['markets'][0]['ticker']
    print(f'Testing orderbook for: {ticker}')

    orderbook = client.get_orderbook(ticker)
    yes_orders = orderbook.get('orderbook', {}).get('yes', [])
    no_orders = orderbook.get('orderbook', {}).get('no', [])

    print(f'YES side: {len(yes_orders)} orders')
    print(f'NO side: {len(no_orders)} orders')

    if yes_orders:
        print(f'Best YES bid: {yes_orders[0][0]}¢')
    if no_orders:
        print(f'Best NO bid: {no_orders[0][0]}¢')
else:
    print('No open markets found')
"
```

**Expected output:**
```
Testing orderbook for: KXAFCCLGAME-24JUN29-B10
YES side: 12 orders
NO side: 8 orders
Best YES bid: 48¢
Best NO bid: 52¢
```

### Test 4: Dry Run Order Placement (Paper Mode)

```bash
# Test order creation without actually sending to Kalshi
python -c "
from rl_bot.kalshi_api import KalshiPaperTradingClient

client = KalshiPaperTradingClient('test', 'test', initial_balance=100.0)

# Place paper order
order = client.create_order(
    ticker='KXAFCCLGAME-TEST',
    side='yes',
    action='buy',
    count=1,
    type='limit',
    yes_price=50
)

print(f'Order ID: {order[\"order_id\"]}')
print(f'Status: {order[\"order\"][\"status\"]}')

# Simulate fill
client.simulate_fill(order['order_id'], fill_price=50, fill_count=1)

# Check position
position = client.get_position_for_ticker('KXAFCCLGAME-TEST')
print(f'Position: {position}')

# Check balance
balance = client.get_balance()
print(f'Balance: \${balance[\"balance\"] / 100:.2f}')
"
```

**Expected output:**
```
[PAPER] Order created: paper_order_1 - buy 1 KXAFCCLGAME-TEST @ 50
Order ID: paper_order_1
Status: resting
[PAPER] Fill executed: buy 1 KXAFCCLGAME-TEST @ 50¢
Position: 1
Balance: $99.50
```

---

## Phase 2: Live API Testing (Minimal Risk)

⚠️ **Only proceed if all Phase 0 and Phase 1 tests pass**

### Test 5: Place ONE Real Order (Smallest Size)

```bash
# Set live mode
export PAPER_MODE=false

# Test placing ONE real order on Kalshi
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os
import time

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

# Find an open market
markets = client.get_markets(series_ticker='KXAFCCLGAME', status='open', limit=1)
if not markets['markets']:
    print('No open markets')
    exit(1)

ticker = markets['markets'][0]['ticker']
print(f'Testing order placement on: {ticker}')

# Get current orderbook
orderbook = client.get_orderbook(ticker)
yes_orders = orderbook.get('orderbook', {}).get('yes', [])

if not yes_orders:
    print('No YES orders in book')
    exit(1)

best_yes_bid = yes_orders[0][0]
print(f'Best YES bid: {best_yes_bid}¢')

# Place a CONSERVATIVE limit order (low probability of fill for testing)
# Use a price well below best bid so it's unlikely to fill immediately
test_price = max(1, best_yes_bid - 10)  # 10 cents below market
print(f'Placing test order at {test_price}¢ (unlikely to fill)')

order = client.place_limit_order(
    ticker=ticker,
    side='buy',
    price_cents=test_price,
    size=1
)

order_id = order.get('order_id')
print(f'✓ Order placed: {order_id}')

# Wait 5 seconds
print('Waiting 5 seconds...')
time.sleep(5)

# Check order status
status = client.get_order(order_id)
print(f'Order status: {status[\"order\"][\"status\"]}')

# Cancel the order
print('Canceling order...')
client.cancel_order(order_id)
print('✓ Order canceled')
"
```

**Expected output:**
```
Testing order placement on: KXAFCCLGAME-24JUN29-B10
Best YES bid: 48¢
Placing test order at 38¢ (unlikely to fill)
✓ Order placed: order_abc123
Waiting 5 seconds...
Order status: resting
Canceling order...
✓ Order canceled
```

**If order fills immediately:** Your test price was too high. This test is to verify order mechanics, not to actually trade.

### Test 6: Full Cycle Test (Order → Fill → Position)

⚠️ **This will execute a real trade. Only do this if you're ready to risk ~$0.50**

```bash
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os
import time

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

# Find market
markets = client.get_markets(series_ticker='KXAFCCLGAME', status='open', limit=1)
ticker = markets['markets'][0]['ticker']

print(f'Full cycle test on: {ticker}')

# Get position before
pos_before = client.get_position_for_ticker(ticker)
print(f'Position before: {pos_before}')

# Place aggressive order (at best bid = likely to fill)
orderbook = client.get_orderbook(ticker)
best_yes_bid = orderbook['orderbook']['yes'][0][0]

print(f'Placing order at market ({best_yes_bid}¢)...')
order = client.place_limit_order(ticker=ticker, side='buy', price_cents=best_yes_bid, size=1)
order_id = order['order_id']

# Wait for fill (up to 30 seconds)
for i in range(30):
    time.sleep(1)
    status = client.get_order(order_id)
    order_status = status['order']['status']
    print(f'  [{i+1}s] Status: {order_status}')

    if order_status == 'executed':
        print('✓ Order filled!')
        break
else:
    print('✗ Order did not fill in 30 seconds')
    client.cancel_order(order_id)
    exit(1)

# Get position after
pos_after = client.get_position_for_ticker(ticker)
print(f'Position after: {pos_after}')
print(f'Change: +{pos_after - pos_before}')

# Close position (sell)
print('Closing position...')
close_order = client.place_limit_order(ticker=ticker, side='sell', price_cents=best_yes_bid, size=1)
time.sleep(5)

pos_final = client.get_position_for_ticker(ticker)
print(f'Final position: {pos_final}')

print('✓ Full cycle test complete')
"
```

**Expected output:**
```
Full cycle test on: KXAFCCLGAME-24JUN29-B10
Position before: 0
Placing order at market (48¢)...
  [1s] Status: resting
  [2s] Status: resting
  [3s] Status: executed
✓ Order filled!
Position after: 1
Change: +1
Closing position...
Final position: 0
✓ Full cycle test complete
```

---

## Phase 3: Integrated Bot Testing (Live Mode, Minimal Capital)

### Test 7: Run Bot in Live Mode (Dry Run)

Modify `live_trader_v2.py` to log orders but not send them:

```python
# In _send_order method, comment out the actual API call:
async def _send_order(self, category: str, ticker: str, side: str, price: int, size: int):
    logger.info(f"ORDER: {category}/{ticker} {side} {size} @ {price}¢")

    # COMMENT OUT FOR DRY RUN:
    # response = self.api_client.place_limit_order(...)

    # Instead, just log:
    logger.info(f"[DRY RUN] Would place order: {side} {size} {ticker} @ {price}¢")
    return  # Don't actually send
```

Run the bot:

```bash
export PAPER_MODE=false  # Use real API for data, but don't send orders
python -m rl_bot.live_trader_v2
```

Monitor for 1 hour:
- [ ] WebSocket connects to real Kalshi
- [ ] Orderbook data received
- [ ] Observations build correctly
- [ ] Model predictions generate
- [ ] Order logic executes (but doesn't send)
- [ ] No crashes

### Test 8: Full Live Bot (Real Orders, 1 Category Only)

⚠️ **Real money at risk. Start with $10-20 allocated to one category.**

1. Modify `live_config_phase1.py` to use only ONE category:

```python
PHASE1_CATEGORIES = [
    CategoryConfig(
        name="KXAFCCLGAME",  # Safest category (reverse overfit)
        test_pnl_per_episode=3.46,
        overfit_ratio=-0.60,
        test_win_rate=0.67,
        max_contracts=1,
        max_inventory=3,
        capital_allocation=15.00,
    ),
]
```

2. Run live:

```bash
export PAPER_MODE=false
python -m rl_bot.live_trader_v2
```

3. Monitor very closely for first hour:
   - Check every 5 minutes
   - Verify fills happening
   - Verify positions tracking correctly
   - Verify PnL updating

4. After 24 hours of successful single-category trading, add second category.

---

## Safety Checklist Before Live Deployment

- [ ] Paper mode ran successfully for 24+ hours
- [ ] All API tests (1-6) passed
- [ ] Test order placed and canceled successfully
- [ ] Full cycle test (order → fill → position) worked
- [ ] Dry run mode showed correct order logic
- [ ] Balance in Kalshi account = $95.82+
- [ ] Risk limits configured ($10 daily loss, -$25 stop loss)
- [ ] Monitoring set up (logs, alerts)
- [ ] Can manually halt bot (know how to kill process)
- [ ] Can manually cancel all orders via Kalshi UI

---

## Emergency Procedures

### Stop Trading Immediately

```bash
# Kill the bot
pkill -f live_trader_v2

# Or find and kill the process
ps aux | grep live_trader_v2
kill <PID>
```

### Cancel All Orders

```bash
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

result = client.cancel_all_orders()
print(f'Canceled all orders: {result}')
"
```

### Check Current Positions

```bash
python -c "
from rl_bot.kalshi_api import KalshiRESTClient
import os

client = KalshiRESTClient(
    api_key=os.getenv('KALSHI_API_KEY'),
    api_secret=os.getenv('KALSHI_API_SECRET')
)

positions = client.get_positions()
for pos in positions['positions']:
    print(f'{pos[\"ticker\"]}: {pos[\"position\"]} contracts')
"
```

---

## Common Issues

### Issue: "No open markets found"

**Cause:** Category has no active markets today
**Fix:** Check Kalshi website, may need to wait for new markets or use different category

### Issue: "WebSocket disconnected"

**Cause:** Network issue or Kalshi downtime
**Fix:** Bot should auto-reconnect (check logs). If not, restart bot.

### Issue: "Order rejected"

**Cause:** Insufficient balance, invalid price, or market closed
**Fix:** Check balance, check market status, verify order params

### Issue: "Fill rate < 5%"

**Cause:** Orders not competitive or low liquidity
**Fix:** Check if orderbook has moved, may need to adjust quoting logic

---

## Success Criteria

Before scaling from 1 category to full 8-category deployment:

| Metric | Target | Status |
|--------|--------|--------|
| **Uptime** | >99% for 1 week | [ ] |
| **Fill rate** | 10-20% | [ ] |
| **Win rate** | 50-70% | [ ] |
| **Daily PnL** | Positive or near $0 | [ ] |
| **No stop-loss triggers** | 0 in 1 week | [ ] |
| **Position tracking** | 100% accurate | [ ] |
| **PnL calculation** | Matches manual calc | [ ] |

**If all criteria met:** Ready to scale to full 8-category deployment!
