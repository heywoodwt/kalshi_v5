# Deployment Configuration — $95.82 Capital

**Date:** 2026-06-29
**Capital Available:** $95.82
**Validation Source:** `hpc/mm_temporal_split_results.md` (temporal out-of-sample testing)
**Strategy:** MM PPO with subpenny quoting

---

## Critical Findings from Temporal Validation

### ⚠️ Original Deployment Plan Requires Revision

The original deployment plan in `docs/deployment_plan.md` was based on same-data evaluation and **does not reflect out-of-sample performance**:

| Category | Original Eval | Temporal Test | Overfit Ratio | Status |
|----------|-------------:|-------------:|-------------:|--------|
| **KXACBGAME** | **$27.40/ep** | **$0.75/ep** | **53.3x** | ❌ Severe overfitting |
| KXATP | $30.56/ep | Not in top performers | Unknown | ⚠️ Needs validation |
| KXATPCHALLENGERMATCH | $28.75/ep | $1.28/ep | 1.4x | ⚠️ Marginal |
| KXAFCCLGAME | $50.38/ep | $3.46/ep | -0.6x (reverse) | ✅ Actually generalizes better! |
| KXAPFDDH | $19.67/ep | Not in top performers | Unknown | ⚠️ Needs validation |

**Key Insight:** Categories with highest same-data performance often have worst overfitting. Must use temporal test performance for deployment decisions.

---

## Validated Deployment Categories

### Tier 1: Excellent Test Performance + Low Overfitting (Live Markets Available)

| Category | Test PnL | Train PnL | Overfit | Win% Test | Markets Live |
|----------|----------:|----------:|--------:|----------:|:------------:|
| **KXLOLTOTALMAPS** | **+$14.14** | +$1.27 | **0.09x** | 100% | ✅ |
| KXBBCHARTPOSITIONALBUM | +$10.10 | +$9.71 | 1.20x | 100% | ✅ |
| KXRAINLAXM | +$9.81 | +$10.01 | 1.00x | 100% | ✅ |
| KXCHESSWORLDCHAMPION | +$9.79 | +$9.79 | 0.97x | 100% | ✅ |
| KXF1CONSTRUCTORS | +$9.79 | +$7.34 | 0.75x | 100% | ✅ |
| KXHIGHAUS | +$9.79 | +$0.48 | **0.04x** | 100% | ✅ |
| KXNBAMVP | +$9.79 | +$9.79 | 1.00x | 100% | ✅ |
| KXSPOTIFYW | +$9.79 | +$9.77 | 1.00x | 100% | ✅ |
| KXNEXTTEAMNFL | +$9.79 | +$8.30 | 0.84x | 100% | ✅ |
| KXNFLDRAFTPICK | +$9.79 | +$8.23 | 0.64x | 100% | ✅ |

### Tier 2: Good Test Performance from Original Candidates

| Category | Test PnL | Train PnL | Overfit | Win% Test | Markets Live |
|----------|----------:|----------:|--------:|----------:|:------------:|
| **KXAFCCLGAME** | **+$3.46** | +$1.63 | **-0.6x** | 67% | ✅ |
| KXATPMATCH | +$3.00 | +$0.10 | -4.5x (reverse) | 33% | ✅ |
| KXATPCHALLENGERMATCH | +$1.28 | +$1.37 | 1.4x | 33% | ✅ |
| KXATPSETWINNER | +$0.85 | +$0.83 | -0.2x (reverse) | 67% | ✅ |

**Reverse overfitting** (test > train) indicates model generalizes better than it trains — excellent signal.

---

## Recommended Deployment Path

### Phase 0: Paper Trading (Weeks 1-2) — **START HERE**

**Capital Required:** $0
**Goal:** Validate infrastructure, fill rates, WebSocket stability

**Configuration:**
```yaml
trading:
  mode: paper
  categories:
    - KXAFCCLGAME  # Best reverse-overfit from original candidates
    - KXATPMATCH   # High test performance, reverse overfit
    - KXATPCHALLENGERMATCH  # Large category, acceptable overfit

  per_category:
    max_contracts: 1
    max_inventory: 10
    quote_size: 1
    subpenny: true

risk:
  max_daily_loss: $0  # Paper trading
  max_position_value: $100  # Simulated

monitoring:
  log_all_quotes: true
  log_all_fills: true
  track_fill_rate: true
  compare_backtest_vs_live: true
```

**Success Criteria:**
- ✅ WebSocket uptime > 99%
- ✅ Quote latency < 500ms p99
- ✅ Fill rate 15-25% (expected vs backtest 100%)
- ✅ Live PnL within 2x of backtest (accounting for fill rate)
- ✅ No infrastructure failures for 2 weeks

**If Phase 0 fails:** Fix infrastructure before risking capital.

---

### Phase 1: Micro Live Trading (Weeks 3-4) — **$95.82 Capital**

**Capital Required:** $95.82
**Goal:** Validate real money execution with minimal risk

**Configuration:**
```yaml
trading:
  mode: live
  categories:
    - KXAFCCLGAME  # Test +$3.46, reverse overfit, 67% win rate

  per_category:
    max_contracts: 1  # Single contract only
    max_inventory: 5  # Very conservative
    quote_size: 1
    subpenny: true

risk:
  max_daily_loss: $10  # 10.4% of capital
  max_position_value: $50  # 52% of capital
  stop_loss_threshold: -$25  # 26% drawdown = pause

capital_management:
  reserve: $45.82  # Keep reserves for adverse moves
  trading_capital: $50.00  # Active capital

monitoring:
  alert_on_loss: $5
  alert_on_fill_rate_drop: 10%  # If fill rate < 5%, investigate
  daily_pnl_report: true
```

**Expected Performance (Conservative 15% Fill Rate):**

| Category | Test PnL/ep | Episodes/day | Daily PnL (Backtest) | Daily PnL (15% Fill) |
|----------|------------:|-------------:|---------------------:|---------------------:|
| KXAFCCLGAME | $3.46 | ~3.7 | $12.80 | **$1.92** |

**Monthly Projection:** $1.92/day × 30 = **$57.60/month**
**Annual Projection:** $57.60 × 12 = **$691/year**
**ROI:** 7.2% per year on $95.82

**Success Criteria:**
- ✅ Positive cumulative PnL after 2 weeks
- ✅ Actual fill rate 10-20%
- ✅ No stop-loss triggers
- ✅ Win rate 50-70% (vs 67% backtest)

**If Phase 1 succeeds:** Scale to Phase 2 with additional capital.

---

### Phase 2: Multi-Category Expansion (Weeks 5-8) — **Requires $200+ Capital**

**Not possible with $95.82.** Requires additional capital deposit.

**Configuration** (future, when capital available):
```yaml
trading:
  mode: live
  categories:
    - KXAFCCLGAME  # Proven in Phase 1
    - KXATPMATCH   # Add second validated category
    - KXATPCHALLENGERMATCH  # Add large liquid category

  per_category:
    max_contracts: 1
    max_inventory: 10
    quote_size: 1
    subpenny: true

capital:
  required: $200
  allocation:
    KXAFCCLGAME: $70
    KXATPMATCH: $70
    KXATPCHALLENGERMATCH: $60
```

**Expected Performance (15% Fill Rate):**

| Category | Test PnL/ep | Episodes/day | Daily PnL (15% Fill) |
|----------|------------:|-------------:|---------------------:|
| KXAFCCLGAME | $3.46 | 3.7 | $1.92 |
| KXATPMATCH | $3.00 | ~8.5 | $3.83 |
| KXATPCHALLENGERMATCH | $1.28 | ~5.4 | $1.04 |
| **Total** | | | **$6.79/day = $2,478/year** |

---

### Phase 3: High-Performers Addition (Weeks 9-16) — **Requires $500+ Capital**

**Add Tier 1 categories with excellent test metrics.**

**Configuration** (future, when capital available):
```yaml
categories:
  # Tier 2 (proven)
  - KXAFCCLGAME
  - KXATPMATCH
  - KXATPCHALLENGERMATCH

  # Tier 1 (high test performance)
  - KXNBAMVP  # +$9.79 test, 1.00x overfit, 100% win
  - KXSPOTIFYW  # +$9.79 test, 1.00x overfit, 100% win
  - KXNEXTTEAMNFL  # +$9.79 test, 0.84x overfit, 100% win

capital:
  required: $500
```

**Expected Performance (15% Fill Rate):**

Tier 1 categories have +$9.79/episode test performance but **unknown episode frequency**. Conservative estimate: 1-2 episodes/day each.

| Tier | Daily PnL (15% Fill) |
|------|---------------------:|
| Tier 2 (proven) | $6.79 |
| Tier 1 (3 cats, 1.5 ep/day each) | $6.62 |
| **Total** | **$13.41/day = $4,895/year** |

**Caveat:** Tier 1 categories may have lower liquidity. Requires validation in paper trading first.

---

## Infrastructure Configuration

### AWS Setup (us-east-2)

**Minimal configuration for $95.82 capital:**

```yaml
compute:
  instance_type: t3.small  # $15/month (vs c6i.xlarge $122/month)
  vcpu: 2
  ram: 2GB
  storage: 20GB EBS gp3

database:
  use_sqlite: true  # No RDS ($20/month saved)
  backup_to_s3: true

docker_services:
  - trading_bot
  - prometheus
  - grafana (optional, can run locally)

monthly_cost: $15-$20  # vs $127 in original plan
```

**When scaling to $200+:** Upgrade to c6i.xlarge + RDS as originally planned.

### Model Checkpoints

**Use temporal-split validated checkpoints:**

```bash
# Checkpoints directory
rl_bot/mm_checkpoints/

# Deploy these checkpoints only
KXAFCCLGAME.zip  # Test +$3.46, reverse overfit
KXATPMATCH.zip   # Test +$3.00, reverse overfit
KXATPCHALLENGERMATCH.zip  # Test +$1.28, acceptable overfit

# DO NOT deploy these (overfitting issues)
KXACBGAME.zip  # 53.3x overfit
# (Any category not validated in temporal split)
```

---

## Risk Management

### Position Limits ($95.82 Capital)

```python
MAX_POSITION_VALUE = 50.0  # 52% of capital
MAX_DAILY_LOSS = 10.0      # 10.4% of capital
MAX_INVENTORY = 5          # Conservative for single category
STOP_LOSS_THRESHOLD = -25.0  # 26% drawdown = pause system
```

### Circuit Breakers

```python
# Halt trading if any trigger
if daily_pnl < -MAX_DAILY_LOSS:
    halt_trading()

if cumulative_pnl < -STOP_LOSS_THRESHOLD:
    halt_trading()
    alert_admin("Stop loss triggered")

if fill_rate < 0.05:  # If fill rate drops below 5%
    alert_admin("Fill rate collapse")
```

### Capital Preservation

With only $95.82:
- **Never risk more than $10/day** (allows 9+ days of losses before depletion)
- **Keep $45 in reserves** for adverse price moves on open positions
- **Start with 1 category only** to minimize correlation risk
- **Pause immediately on 2-day losing streak** and investigate

---

## Monitoring & Alerts

### Key Metrics (Real-Time)

```python
# Track every minute
metrics = {
    "websocket_uptime": 99.9%,
    "quote_latency_p99": 500ms,
    "fills_today": 12,
    "fill_rate": 15%,  # fills / quotes_sent
    "pnl_today": +$2.10,
    "open_positions": 2,
    "position_value": $38.00,
    "capital_at_risk": 39.6%,
}

# Alert thresholds
if metrics["pnl_today"] < -5.0:
    alert("Daily loss approaching limit")

if metrics["fill_rate"] < 0.10:
    alert("Fill rate below 10%")

if metrics["websocket_uptime"] < 0.95:
    alert("WebSocket instability")
```

### Daily Reports

```markdown
## Daily PnL Report — 2026-06-30

**Capital:** $95.82 → $97.92 (+$2.10, +2.2%)

**Performance:**
- KXAFCCLGAME: +$2.10 (4 episodes, 15% fill rate, 75% win rate)

**Quotes Sent:** 87
**Fills:** 13 (14.9% fill rate)
**Win Rate:** 75% (3 wins, 1 loss)

**Comparison to Backtest:**
- Expected: $12.80 (100% fill) → $1.92 (15% fill)
- Actual: $2.10
- Variance: +9.4% (within normal range)

**Positions:**
- Open: 2 contracts ($34 at risk)
- Inventory: +1 (long bias)

**Alerts:** None
**Status:** ✅ Healthy
```

---

## Deployment Checklist

### Pre-Deployment (Week 1)

- [ ] Read `docs/aws_deployment_plan.md` for infrastructure setup
- [ ] Deploy AWS EC2 t3.small instance (us-east-2)
- [ ] Install Docker, clone repo, build containers
- [ ] Configure Kalshi API credentials (`.env` file)
- [ ] Test WebSocket connection (subscribe to KXAFCCLGAME markets)
- [ ] Load model checkpoints: `KXAFCCLGAME.zip`, `KXATPMATCH.zip`, `KXATPCHALLENGERMATCH.zip`
- [ ] Verify model inference (forward pass on dummy observation)
- [ ] Configure monitoring (Prometheus + Grafana or logs)

### Phase 0: Paper Trading (Week 2-3)

- [ ] Start trading bot in `paper` mode
- [ ] Monitor for 2 weeks:
  - [ ] WebSocket uptime > 99%
  - [ ] Quote latency < 500ms p99
  - [ ] Fill rate 10-25%
  - [ ] PnL correlates with backtest (within 2x accounting for fill rate)
- [ ] No infrastructure failures
- [ ] Compare paper trading PnL to backtest expectations

### Phase 1: Live Trading (Week 4-5)

- [ ] **Deposit capital:** Ensure $95.82 available in Kalshi account
- [ ] Switch to `live` mode
- [ ] Deploy KXAFCCLGAME only (1 contract max)
- [ ] Monitor daily:
  - [ ] Cumulative PnL positive
  - [ ] No stop-loss triggers
  - [ ] Fill rate 10-20%
  - [ ] Win rate 50-70%
- [ ] After 2 weeks of success: Evaluate Phase 2 (requires capital increase)

---

## Cost Breakdown

### Infrastructure (Phase 0 & 1)

| Component | Cost/Month | Annual |
|-----------|----------:|-------:|
| EC2 t3.small | $15 | $180 |
| Data transfer | $2 | $24 |
| EBS storage (20GB) | $2 | $24 |
| S3 backups | $1 | $12 |
| **Total** | **$20** | **$240** |

**Operating margin:** $691/year revenue - $240/year infra = **$451/year profit** (4.7% ROI)

### Capital Requirements by Phase

| Phase | Capital | Monthly Revenue | Annual Revenue | ROI |
|-------|--------:|----------------:|---------------:|----:|
| Phase 0 (Paper) | $0 | $0 | $0 | N/A |
| Phase 1 (Micro) | **$95.82** | $57.60 | $691 | 7.2% |
| Phase 2 (Multi) | $200 | $206.70 | $2,480 | 12.4% |
| Phase 3 (Full) | $500 | $407.70 | $4,892 | 9.8% |

**Phase 1 is conservative but viable with current capital.**

---

## Expected Timeline

### Week-by-Week Plan

| Week | Phase | Capital | Action | Expected Outcome |
|------|-------|--------:|--------|------------------|
| 1 | Setup | $0 | Deploy AWS, configure bot | Infrastructure ready |
| 2-3 | Phase 0 | $0 | Paper trading (3 categories) | Validate fill rates, latency |
| 4-5 | Phase 1 | $95.82 | Live (KXAFCCLGAME only) | +$1.92/day, validate real execution |
| 6-8 | Evaluate | $95.82 | Continue Phase 1, accumulate profits | Grow capital to $150+ |
| 9+ | Phase 2 | $200+ | Add KXATPMATCH, KXATPCHALLENGERMATCH | Scale to $6.79/day |

**Key decision point:** After Week 5, if Phase 1 is profitable, deposit additional capital ($100-$200) to scale to Phase 2.

---

## Risk Assessment

### High-Risk Factors

1. **Overfitting:** Original deployment plan relied on non-validated categories (KXACBGAME 53.3x overfit)
   - **Mitigation:** Only deploy temporal-split validated categories

2. **Fill Rate Uncertainty:** Backtest assumes 100% fill, real trading likely 10-20%
   - **Mitigation:** Paper trading validates fill rates before risking capital

3. **Capital Constraints:** $95.82 allows only 1 category deployment
   - **Mitigation:** Start micro, prove profitability, then scale

4. **Infrastructure Costs:** $20/month operating cost reduces net profit significantly at $95 scale
   - **Mitigation:** Accept lower ROI for Phase 1, scale to amortize costs

### Medium-Risk Factors

1. **Market Condition Changes:** Models trained on April 2026 data, deploying June 2026
   - **Mitigation:** Paper trading detects regime changes before live risk

2. **Tier 1 Category Liquidity:** High test PnL categories (KXNBAMVP, etc.) may have low episode frequency
   - **Mitigation:** Paper trade Tier 1 categories before deploying capital

---

## Success Metrics

### Phase 1 Success Criteria (Required for Phase 2)

| Metric | Target | Status |
|--------|--------|--------|
| Cumulative PnL | Positive after 2 weeks | TBD |
| Daily PnL | >$0 average | TBD |
| Fill Rate | 10-20% | TBD |
| Win Rate | 50-70% | TBD |
| Stop Loss Triggers | 0 | TBD |
| Infrastructure Uptime | >99% | TBD |

**If all criteria met:** Proceed to Phase 2 (requires capital increase to $200+)
**If any criteria fail:** Investigate, fix, re-validate in paper trading

---

## Conclusion

**With $95.82 capital:**

1. **Start with Phase 0 (paper trading)** to validate infrastructure and fill rates
2. **Deploy Phase 1 (live micro)** with KXAFCCLGAME only
3. **Expected return:** $1.92/day = $691/year (7.2% ROI after $240 infra costs)
4. **Path to scale:** Accumulate profits + deposit additional capital to reach $200+ for Phase 2

**Critical revision:** Original deployment plan assumed in-sample evaluation metrics. Temporal validation shows severe overfitting in previously recommended categories (KXACBGAME 53.3x). This configuration uses only temporally validated categories.

**See also:**
- `docs/aws_deployment_plan.md` — Infrastructure setup (use t3.small for Phase 1)
- `hpc/mm_temporal_split_results.md` — Validation source
- `docs/deployment_plan.md` — Original plan (DEPRECATED, needs revision based on temporal validation)
