# Phase 1 Live Deployment Guide

**Capital:** $95.82
**Categories:** 8 temporally validated markets
**Expected Return:** $12/day = $360/month = $4,381/year (45.7% ROI)

---

## Pre-Deployment Checklist

### 1. Verify Checkpoints

Ensure you have model checkpoints for all 8 categories:

```bash
ls -lh rl_bot/mm_checkpoints/

# Required checkpoints
KXLOLTOTALMAPS.zip
KXBBCHARTPOSITIONALBUM.zip
KXRAINLAXM.zip
KXNBAMVP.zip
KXSPOTIFYW.zip
KXAFCCLGAME.zip
KXATPMATCH.zip
KXATPCHALLENGERMATCH.zip
```

**If missing:** Download from HPC:

```bash
rsync -avz mtk9va@login.hpc.virginia.edu:/scratch/mtk9va/kalshi_v5/rl_bot/mm_checkpoints/ rl_bot/mm_checkpoints/
```

### 2. Verify Capital

Ensure your Kalshi account has **$95.82+** available:

```bash
# Check balance via Kalshi REST API (need to implement)
# Or check manually at https://kalshi.com/account
```

### 3. Set Environment Variables

Create `.env` file:

```bash
cat > .env <<EOF
KALSHI_API_KEY=your_api_key_here
KALSHI_API_SECRET=your_api_secret_here
KALSHI_BASE_URL=https://trading-api.kalshi.com
KALSHI_WS_URL=wss://trading-api.kalshi.com/trade-api/ws/v2
EOF
```

**Security:** Never commit `.env` to git!

### 4. Verify Market Availability

Check that all 8 categories have open markets:

```bash
# TODO: Script to check market status via Kalshi API
python rl_bot/verify_markets.py
```

Expected: All 8 categories have `status=open` markets.

---

## Deployment Options

### Option A: AWS Deployment (Recommended)

**Cost:** $15-20/month
**Uptime:** 24/7
**Latency:** Low (us-east-2)

See `docs/aws_deployment_plan.md` for full AWS setup.

**Quick AWS Deploy:**

```bash
# Launch EC2 t3.small (us-east-2)
# SSH into instance
ssh -i ~/.ssh/kalshi-key.pem ubuntu@<ec2-ip>

# Clone repo
git clone https://github.com/yourusername/kalshi_v5.git
cd kalshi_v5

# Install dependencies
pip install -r requirements.txt

# Copy .env file
scp .env ubuntu@<ec2-ip>:~/kalshi_v5/.env

# Run trader
nohup python -m rl_bot.live_trader > trader.log 2>&1 &

# Monitor
tail -f trader.log
tail -f live_trading.log
```

### Option B: Local Deployment

**Cost:** $0 (uses your machine)
**Uptime:** Only when your machine is on
**Latency:** Depends on your internet

**Run locally:**

```bash
# From project root
python -m rl_bot.live_trader
```

**Keep running in background:**

```bash
nohup python -m rl_bot.live_trader > trader.log 2>&1 &

# Or use tmux/screen
tmux new -s trader
python -m rl_bot.live_trader
# Ctrl+B, D to detach
```

---

## Configuration Review

Run config script to verify setup:

```bash
python rl_bot/live_config_phase1.py
```

**Expected output:**

```
================================================================================
PHASE 1 LIVE TRADING CONFIGURATION
================================================================================

Capital: $95.82
Active: $90.00
Reserve: $5.82
Categories: 8

Category Allocations:
--------------------------------------------------------------------------------
KXLOLTOTALMAPS            $12.00     Test: $14.14    Overfit: 0.09  x  Win%: 100  %
KXBBCHARTPOSITIONALBUM    $10.00     Test: $10.10    Overfit: 1.20  x  Win%: 100  %
KXRAINLAXM                $10.00     Test: $9.81     Overfit: 1.00  x  Win%: 100  %
KXNBAMVP                  $10.00     Test: $9.79     Overfit: 1.00  x  Win%: 100  %
KXSPOTIFYW                $10.00     Test: $9.79     Overfit: 1.00  x  Win%: 100  %
KXAFCCLGAME               $15.00     Test: $3.46     Overfit: -0.60 x  Win%: 67   %
KXATPMATCH                $13.00     Test: $3.00     Overfit: -4.50 x  Win%: 33   %
KXATPCHALLENGERMATCH      $10.00     Test: $1.28     Overfit: 1.40  x  Win%: 33   %


Expected Performance (15% Fill Rate):
-------------------------------------------------------------------------------------
TOTAL                                           $80.02          $12.00

Monthly: $360.07
Annual: $4380.88
ROI: 45.7% ($4380.88 profit on $95.82 capital)

Risk Limits:
  Max daily loss: $10.00
  Stop loss threshold: $-25.00
  Max position value: $60.00
```

---

## Launch Sequence

### Step 1: Dry Run (Recommended First)

Test the bot without real orders:

```bash
# Edit live_trader.py:
# Change _send_order() to only log, not execute

python -m rl_bot.live_trader
```

Monitor for 1 hour:
- ✅ WebSocket connects
- ✅ Models load successfully
- ✅ Observations build correctly
- ✅ Actions generate (logged only)
- ✅ No errors/crashes

### Step 2: Live Launch

```bash
# Ensure _send_order() calls actual Kalshi REST API
# (Need to implement Kalshi REST API integration)

python -m rl_bot.live_trader
```

**First 10 minutes:** Watch closely
- Verify quotes are sent
- Verify fills are happening
- Check PnL updates

**First hour:**
- Monitor fill rate (expect 10-20%)
- Check for errors
- Verify risk limits work

**First day:**
- Daily PnL should be positive or near $0
- No stop-loss triggers
- Fill rate 10-20%

---

## Monitoring

### Real-Time Monitoring

```bash
# Tail logs
tail -f live_trading.log

# Check status (every 5 minutes)
grep "PnL" live_trading.log | tail -20

# Check fills
grep "FILL" live_trading.log | wc -l

# Check quotes
grep "ORDER" live_trading.log | wc -l
```

### Daily Reports

Bot generates daily reports:

```
## Daily PnL Report — 2026-06-30

**Capital:** $95.82 → $107.82 (+$12.00, +12.5%)

**Performance:**
- KXLOLTOTALMAPS: +$1.06 (0.5 episodes, 15% fill rate, 100% win)
- KXBBCHARTPOSITIONALBUM: +$1.51 (1.0 episodes, 15% fill rate, 100% win)
- KXAFCCLGAME: +$1.92 (3.7 episodes, 15% fill rate, 67% win)
- KXATPMATCH: +$3.82 (8.5 episodes, 15% fill rate, 33% win)
- ...

**Quotes Sent:** 347
**Fills:** 52 (15.0% fill rate)
**Overall Win Rate:** 62%

**Alerts:** None
**Status:** ✅ Healthy
```

### Alerts

Bot will alert on:
- Daily loss > $5
- Stop loss triggered ($-25)
- Fill rate < 5%
- WebSocket downtime > 60 seconds

**Set up email/SMS alerts:**

```bash
# TODO: Implement alerting (Twilio, email, etc.)
```

---

## Risk Management

### Automatic Halts

Trading halts automatically if:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Daily loss | -$10.00 | Halt all trading for the day |
| Stop loss | -$25.00 | Halt all trading permanently, alert admin |
| Position value | >$60.00 | Stop new orders until positions reduce |
| Consecutive losses | 3 in a row per category | Pause that category |

### Manual Halt

```bash
# Emergency stop
pkill -f live_trader

# Or send SIGTERM
kill $(pgrep -f live_trader)

# Cancel all open orders
# TODO: Script to cancel all orders via Kalshi API
```

---

## Troubleshooting

### WebSocket Disconnects

**Symptom:** `WebSocket connection closed` in logs

**Fix:**
1. Bot should auto-reconnect (TODO: implement reconnect logic)
2. If not, restart bot: `pkill -f live_trader && python -m rl_bot.live_trader`

### Low Fill Rate (<5%)

**Symptom:** Very few fills despite many quotes

**Possible causes:**
- Kalshi liquidity dried up (check manually)
- Quote prices not competitive (inspect orderbook)
- Account issue (check Kalshi account status)

**Fix:**
- Monitor for 1 day
- If <5% persists, investigate specific categories

### High Losses

**Symptom:** Daily PnL < -$5

**Action:**
1. Check if specific category is losing
2. Review that category's fills (were they adverse?)
3. If one category problematic, remove it from config
4. If all categories negative, halt and investigate

---

## Scaling to Phase 2

After 2-4 weeks of successful Phase 1:

**Success criteria:**
- ✅ Cumulative PnL > $0
- ✅ Average daily PnL > $10
- ✅ Fill rate 10-20%
- ✅ Win rate 50-70%
- ✅ No stop-loss triggers

**Phase 2 requires $200+ capital:**
1. Deposit additional $100-150
2. Add more categories from top 262 validated markets
3. Increase per-category allocations
4. Scale to $20-30/day expected

---

## Important Notes

### Kalshi API Integration (TODO)

The current `live_trader.py` script has **placeholder** Kalshi REST API calls. You need to:

1. Implement `_send_order()` to actually place orders via Kalshi REST API
2. Implement order status tracking (fills, cancels)
3. Implement position tracking from Kalshi account
4. Implement PnL calculation from actual fills

**Kalshi API docs:** https://trading-api.readme.io/

### Testing Before Live

**CRITICAL:** Test on Kalshi's **demo environment** first if available, or:

1. Run dry-run mode (orders logged, not sent)
2. Start with smallest possible size (1 contract only)
3. Monitor very closely for first hour/day
4. Only scale after confirming system works

---

## Contact / Alerts

Set up alerts to notify you:

- **Daily:** Summary report emailed
- **Immediate:** SMS for stop-loss or major errors

```python
# TODO: Implement in live_trader.py
def send_alert(message):
    # Email via SendGrid
    # SMS via Twilio
    # Slack webhook
    pass
```

---

## Deployment Checklist

- [ ] Verify all 8 model checkpoints exist
- [ ] Verify $95.82+ balance in Kalshi account
- [ ] Set up `.env` with API credentials
- [ ] Verify all 8 categories have open markets
- [ ] Test dry-run mode (1 hour)
- [ ] Deploy to AWS or run locally
- [ ] Monitor first 10 minutes closely
- [ ] Check daily report after day 1
- [ ] Review weekly performance after week 1
- [ ] Scale to Phase 2 after 2-4 weeks of success

**Good luck! 🚀**
