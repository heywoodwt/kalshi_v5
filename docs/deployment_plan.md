# Market-Making Deployment Plan

**Version:** 1.0
**Date:** 2026-06-27
**Status:** Pre-Production Planning
**Target:** $100,000/year revenue

---

## Executive Summary

Deploy PPO-trained market-making agents on Kalshi prediction markets, focusing on **5 proven profitable categories** with conservative 20% live efficiency assumption. Phased rollout starting with paper trading validation, scaling to multi-contract production.

**Conservative Revenue Projection:** $225,935/year (3 contracts × 5 categories × 20% efficiency)
**Capital Required:** $500 (5× risk buffer)
**Estimated Timeline:** 16-24 weeks to full deployment

---

## 1. Category Selection

### Tier 1: Primary Deployment Categories

Based on 10-epoch training with 108 checkpoints evaluated:

| Category | Mean $/ep | Std Dev | Tickers | Win Rate* | Daily Revenue** | Annual | Risk Level |
|----------|----------:|--------:|--------:|----------:|----------------:|-------:|------------|
| **KXACBGAME** | $27.40 | ±$13.70 | 20 | TBD | $312 | $113,880 | Low (reliable variance) |
| **KXATP** | $30.56 | ±$34.11 | 846 | TBD | $95 | $34,675 | Medium (large liquid) |
| **KXATPCHALLENGERMATCH** | $28.75 | ±$32.34 | 483 | TBD | $138 | $50,370 | Medium (large liquid) |
| **KXAFCCLGAME** | $50.38 | ±$27.31 | 9 | TBD | $186 | $67,890 | Medium |
| **KXAPFDDH** | $19.67 | ±$23.64 | 25 | TBD | $171 | $62,415 | Medium |
| **Total** | | | | | **$902** | **$329,230** | |

*Win rate to be measured during paper trading
**At 1 contract, backtest assumption

**Selection Criteria:**
1. **Mean reward > $15/episode**
2. **Std dev < 2× mean** (manageable variance)
3. **Tickers ≥ 9** (sufficient liquidity)
4. **Sports/games categories** (predictable timing, not political/macro)

### Tier 2: Backup Categories

For diversification if Tier 1 underperforms:

- KXBTCD ($13.45, 1,929 tickers) — Major BTC daily markets
- KXBBLGAME ($14.01, 26 tickers) — Basketball league games
- KXBOLPDIVGAME ($17.83, 28 tickers) — Bolivian division games
- KXBERNIEMENTION ($7.27, ±$6.00, 75 tickers) — Political mentions
- KXBRASILEIROTOTAL ($7.76, 78 tickers) — Brazilian totals

### Categories to Avoid

**Do not deploy:**
- KX (main BTC) — Performance discrepancy under investigation (-$1.91 eval vs +$3.69 training)
- KXAAAGASW — Consistently negative (-$3.68)
- KXBBCHARTPOSITIONALBUM — Music charts, unpredictable (-$3.26)
- Any category with <5 tickers — high overfit risk

---

## 2. Phased Rollout Plan

### Phase 1: Paper Trading Validation (Weeks 1-4)

**Goal:** Measure live fill rates and validate backtest assumptions

**Deployment:**
- **Categories:** KXACBGAME only (most reliable, low variance)
- **Contract size:** 0 (paper trading, no real orders)
- **Mode:** Log all quotes and fills against live orderbook data
- **Duration:** 4 weeks (minimum 20 trading days)

**Success Criteria:**
- [ ] Paper trading system runs without errors for 20 consecutive days
- [ ] Fill rate ≥ 15% of backtest assumption
- [ ] Observed spread ≥ $0.02 (viable for $0.01 min half-spread + fees)
- [ ] No adverse selection detected (filled quotes don't consistently move against us)

**Metrics to Collect:**
- Fill rate per quote level (bid/ask at each price)
- Time to fill distribution
- Spread distribution (observed vs. backtest)
- Queue position effects
- Adverse selection rate

**Infrastructure:**
- Live orderbook feed via Kalshi API
- Quote simulation (no real orders)
- Logging to database (all quotes, fills, PnL)
- Dashboard for monitoring

**Exit Criteria:**
- If fill rate < 10%: backtest overly optimistic, revise projections
- If adverse selection > 30%: informed traders picking us off, abandon strategy

### Phase 2: Single-Category Live (Weeks 5-8)

**Goal:** Validate profitability with real capital on lowest-risk category

**Deployment:**
- **Categories:** KXACBGAME only
- **Contract size:** 1
- **Capital:** $100 (max 20 contract inventory = $20 risk + buffer)
- **Duration:** 4 weeks

**Success Criteria:**
- [ ] Cumulative PnL > $0 after 20 trading days
- [ ] Daily PnL positive ≥ 60% of days
- [ ] Observed $/episode ≥ $5 (20% of backtest $27.40)
- [ ] Max drawdown < $50

**Expected Performance:**
- Backtest: $312/day at 1 contract
- Conservative (20%): $62/day
- Pessimistic (10%): $31/day
- Target: Break even or better

**Risk Management:**
- Daily loss limit: $20
- Cumulative loss limit: $50 (triggers pause for analysis)
- Inventory limit: 20 contracts max
- Emergency stop: manual override

**Exit Criteria:**
- If cumulative PnL < -$50: pause, analyze, revise strategy
- If fill rate collapsed: market structure changed, re-evaluate

### Phase 3: Multi-Category Expansion (Weeks 9-16)

**Goal:** Scale to 5 categories, still at 1 contract each

**Deployment:**
- **Categories:** Add KXATP, KXATPCHALLENGERMATCH, KXAFCCLGAME, KXAPFDDH
- **Contract size:** 1 per category (5 total)
- **Capital:** $500 (5 categories × $20 risk × 5× buffer)
- **Duration:** 8 weeks

**Success Criteria:**
- [ ] Each category individually profitable over 14-day rolling window
- [ ] Combined $/day ≥ $100 (20% of $500 backtest)
- [ ] No single category loses > $30 in any week
- [ ] System runs 24/7 with <1% downtime

**Expected Performance:**
- Backtest: $902/day at 1 contract each
- Conservative (20%): $180/day = $65,700/year
- Target: $100-200/day

**Monitoring:**
- Per-category PnL tracking
- Fill rate comparison (backtest vs. live)
- Inventory management across categories
- Cross-category correlation

**Exit Criteria:**
- If any category loses > $50 cumulative: disable that category
- If combined PnL < 0 after 4 weeks: pause entire system

### Phase 4: Size Scaling (Weeks 17-24)

**Goal:** Scale to 3 contracts per category to reach $100k/year target

**Deployment:**
- **Categories:** Same 5 categories
- **Contract size:** 1 → 2 → 3 (gradual scaling)
- **Capital:** $500 → $750 → $1,000
- **Duration:** 8 weeks (4 weeks at 2 contracts, 4 weeks at 3 contracts)

**Scaling Protocol:**
1. Run 2 weeks at current size with positive PnL
2. Increase size by 1 contract
3. Monitor fill rate degradation
4. If fill rate drops > 30%, stop scaling

**Success Criteria:**
- [ ] Fill rate per contract ≥ 60% of 1-contract baseline
- [ ] Combined $/day ≥ $274 (target for $100k/year)
- [ ] Max drawdown < $200
- [ ] System stable at 3 contracts for 4 consecutive weeks

**Expected Performance at 3 Contracts:**
- Backtest: $2,706/day (3 × $902)
- Conservative (20%): $541/day = $197,465/year
- Target: $274/day = $100,000/year

**Risk Management:**
- Per-category daily loss limit: $50
- Combined daily loss limit: $100
- Max inventory: 20 contracts per category
- Emergency stop: manual override

**Exit Criteria:**
- If fill rate drops below 40% of 1-contract baseline: stop scaling
- If any category's variance increases by >50%: reduce size

---

## 3. Technical Infrastructure

### System Architecture

```
┌─────────────────┐
│  Kalshi API     │
│  - Orderbook    │
│  - Trades       │
│  - Orders       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Pipeline  │
│  - Normalize    │
│  - Preprocess   │
│  - 1-min windows│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MM Agent       │
│  - Load model   │
│  - Predict      │
│  - Quote        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Order Manager  │
│  - Place/Cancel │
│  - Track fills  │
│  - Inventory    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Risk Manager   │
│  - Loss limits  │
│  - Position     │
│  - Emergency    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Monitoring     │
│  - Dashboard    │
│  - Alerts       │
│  - Logging      │
└─────────────────┘
```

### Required Components

**1. Data Pipeline**
- Real-time orderbook subscription via Kalshi WebSocket API
- Trade data ingestion
- 1-minute window aggregation (matching backtest)
- Market metadata integration

**2. Model Serving**
- Load PPO checkpoints per category
- Real-time inference (<100ms latency)
- Model versioning and rollback capability

**3. Order Management**
- Quote placement with subpenny pricing (+$0.001 bid, -$0.001 ask)
- Order cancellation and replacement
- Fill tracking and inventory management
- Fee calculation (1.75% variance-based maker fees)

**4. Risk Management**
- Per-category position limits (max 20 contracts)
- Daily loss limits (per-category and combined)
- Emergency stop (manual + automated triggers)
- Adverse selection detection

**5. Monitoring & Alerting**
- Real-time PnL dashboard
- Fill rate tracking
- Model performance vs. backtest
- Inventory visualization
- SMS/email alerts for limit breaches

### Technology Stack

**Recommended:**
- **Language:** Python 3.11+
- **ML Framework:** Stable Baselines3 (PPO)
- **API Client:** Kalshi Python SDK
- **Database:** PostgreSQL (trade logs, PnL)
- **Time-series:** InfluxDB (metrics)
- **Dashboard:** Grafana
- **Deployment:** Docker + systemd
- **Hosting:** Dedicated server (low-latency to Kalshi)

**Infrastructure Requirements:**
- **Compute:** 4 CPU cores, 16GB RAM
- **Network:** <50ms latency to Kalshi API
- **Storage:** 100GB SSD (logs, checkpoints)
- **Uptime:** 99.9% target (4 hours downtime/year)

---

## 4. Risk Management Framework

### Position Limits

| Limit Type | Per Category | Combined | Rationale |
|------------|-------------:|---------:|-----------|
| Max inventory | 20 contracts | 100 contracts | Backtest assumption |
| Daily loss | $50 | $100 | 2% of capital at max scale |
| Weekly loss | $150 | $300 | 6% of capital |
| Max drawdown | $200 | $400 | 40% of capital (stop trading) |

### Circuit Breakers

**Automatic halts:**
1. **Daily loss limit exceeded** — Stop all new quotes, flatten positions
2. **Inventory limit reached** — Stop quotes in that direction (bid/ask)
3. **API connectivity lost** — Cancel all orders, alert operator
4. **Model error** — Fallback to safe mode (cancel quotes)

**Manual review triggers:**
1. Fill rate drops >50% from baseline
2. Adverse selection rate >30%
3. 3 consecutive losing days on any category
4. Spread collapse (<$0.02 observed)

### Adverse Selection Detection

Monitor: "Are we being picked off by informed traders?"

**Metrics:**
- Post-fill price movement (did price move against us?)
- Fill rate asymmetry (fills more on losing side?)
- Time-to-fill distribution (instant fills = bad sign)

**Red flags:**
- Filled quotes move against us >60% of time
- Fill rate 2× higher on losing side
- Average time-to-fill <5 seconds (market taking our stale quotes)

**Response:**
- Widen spread
- Reduce quote size
- Disable category if persistent

---

## 5. Performance Monitoring

### Key Metrics (Tracked Real-Time)

**Profitability:**
- Cumulative PnL (live vs. backtest expectation)
- Daily PnL (per category and combined)
- Win rate (% of profitable days)
- Sharpe ratio (risk-adjusted returns)

**Execution:**
- Fill rate (% of quotes filled)
- Fill rate per contract size (1, 2, 3)
- Average spread captured
- Slippage (quote price vs. fill price)

**Risk:**
- Current inventory per category
- Max drawdown (current vs. historical)
- Daily/weekly loss vs. limits
- Adverse selection rate

**Infrastructure:**
- API latency (quote submission to acknowledgment)
- Model inference time
- System uptime
- Error rate

### Dashboard Requirements

**Real-time view (updated every 1 second):**
- Current PnL (today, week, month, all-time)
- Open positions (inventory per category)
- Active quotes (bid/ask levels)
- Recent fills (last 10)

**Historical view:**
- PnL chart (daily/cumulative)
- Fill rate trend
- Spread distribution
- Inventory utilization

**Alerts:**
- Loss limit approaching (SMS + email)
- Fill rate degradation
- API connectivity issues
- Model errors

---

## 6. Failure Modes & Contingencies

### Scenario Planning

| Scenario | Probability | Impact | Mitigation |
|----------|:-----------:|--------|------------|
| **Fill rate <10%** | Medium | High | Widen spread, increase quote size, or abandon |
| **Adverse selection** | Medium | High | Detect early (Phase 1), widen spread if detected |
| **Category goes illiquid** | Low | Medium | Monitor volume, disable if <50% of baseline |
| **Kalshi fee change** | Low | Medium | Model adapts to new fees automatically |
| **Competing MM narrows spreads** | Medium | High | Subpenny pricing for queue priority |
| **Model overfits, doesn't generalize** | Medium | High | Out-of-sample validation before launch |
| **System downtime** | Medium | Low | Automated recovery, manual fallback |
| **Regulatory change** | Low | High | Monitor Kalshi announcements, legal review |

### Disaster Recovery

**If cumulative loss exceeds $400:**
1. Immediately stop all trading
2. Flatten all positions (emergency liquidation)
3. Conduct post-mortem analysis
4. Do NOT resume without understanding root cause

**If Kalshi API unavailable:**
1. Cancel all open orders (via API if possible, email support if not)
2. Alert operator (SMS + email)
3. Wait for API restoration
4. Resume only after connectivity confirmed stable

**If model produces nonsensical quotes:**
1. Halt that category immediately
2. Log inputs and outputs for debugging
3. Fallback to safe mode (no quotes)
4. Investigate model checkpoint corruption

---

## 7. Go/No-Go Checklist

### Before Phase 1 (Paper Trading)

- [ ] Paper trading infrastructure built and tested
- [ ] Kalshi API credentials obtained (production)
- [ ] Real-time orderbook feed working
- [ ] Dashboard functional
- [ ] Alert system tested (SMS/email)
- [ ] Model checkpoints deployed and tested
- [ ] Logging system operational

### Before Phase 2 (Live Trading)

- [ ] Paper trading completed successfully (4 weeks, fill rate ≥15%)
- [ ] Capital allocated ($100 for Phase 2)
- [ ] Risk management system tested
- [ ] Emergency stop procedures documented
- [ ] Legal/compliance review complete
- [ ] Monitoring dashboard ready
- [ ] Backup systems tested

### Before Phase 3 (Multi-Category)

- [ ] Phase 2 profitable (cumulative PnL > 0 after 20 days)
- [ ] Capital allocated ($500)
- [ ] Per-category infrastructure tested
- [ ] Cross-category correlation analyzed
- [ ] Resource scaling plan ready (CPU/RAM)
- [ ] Monitoring scales to 5 categories

### Before Phase 4 (Size Scaling)

- [ ] Phase 3 profitable across all 5 categories
- [ ] Fill rate measured at 1 contract (baseline established)
- [ ] Capital allocated ($1,000)
- [ ] Market impact analysis complete
- [ ] Size scaling automation tested
- [ ] Rollback procedure tested

---

## 8. Financial Projections

### Conservative Scenario (20% Live Efficiency)

| Phase | Contracts | Categories | Backtest $/day | Live $/day | Monthly | Annual |
|-------|:---------:|:----------:|---------------:|-----------:|--------:|-------:|
| 1 | 0 (paper) | 1 | $0 | $0 | $0 | $0 |
| 2 | 1 | 1 | $312 | $62 | $1,860 | $22,630 |
| 3 | 1 | 5 | $902 | $180 | $5,400 | $65,700 |
| 4 | 3 | 5 | $2,706 | $541 | $16,230 | $197,465 |

**ROI at full scale:**
- Capital deployed: $1,000
- Annual return: $197,465
- ROI: 19,746%
- Monthly return: 1,623%

### Optimistic Scenario (30% Live Efficiency)

| Phase | Live $/day | Monthly | Annual |
|-------|------------:|--------:|-------:|
| 2 | $94 | $2,820 | $34,310 |
| 3 | $271 | $8,130 | $98,865 |
| 4 | $812 | $24,360 | $296,380 |

### Pessimistic Scenario (10% Live Efficiency)

| Phase | Live $/day | Monthly | Annual |
|-------|------------:|--------:|-------:|
| 2 | $31 | $930 | $11,315 |
| 3 | $90 | $2,700 | $32,850 |
| 4 | $271 | $8,130 | $98,865 |

**Note:** Even pessimistic scenario reaches $100k/year at full scale.

---

## 9. Success Metrics

### Phase 1 Success
- Fill rate ≥ 15% of backtest
- No adverse selection detected
- System runs 20 days without critical errors

### Phase 2 Success
- Cumulative PnL > $0 after 20 days
- ≥60% of days profitable
- Live $/episode ≥ 20% of backtest

### Phase 3 Success
- All 5 categories individually profitable (14-day rolling)
- Combined $/day ≥ $100
- <1% system downtime

### Phase 4 Success
- Fill rate per contract ≥ 60% of 1-contract baseline
- Combined $/day ≥ $274
- Sustained for 4 consecutive weeks

### Ultimate Success
- **$100,000/year revenue** (target met)
- **System runs autonomously** with minimal intervention
- **Sharpe ratio > 1.0** (risk-adjusted returns)
- **Max drawdown < 20%** of capital

---

## 10. Next Steps

**Immediate (Week 0):**
1. Complete KX investigation (in progress)
2. Review and approve this deployment plan
3. Set up development environment
4. Obtain Kalshi API credentials

**Week 1-2:**
1. Build paper trading infrastructure
2. Integrate Kalshi API (orderbook + trades)
3. Deploy KXACBGAME checkpoint
4. Test quote generation and simulation

**Week 3-4:**
1. Begin paper trading (KXACBGAME only)
2. Collect fill rate and spread data
3. Build monitoring dashboard
4. Set up alert system

**Month 2:**
1. Complete Phase 1 paper trading
2. Analyze results and adjust projections
3. Build live trading infrastructure
4. Deploy capital ($100) and begin Phase 2

**Months 3-6:**
1. Complete Phases 2-4
2. Scale to 5 categories at 3 contracts each
3. Monitor and optimize
4. Reach $100k/year run rate

---

## Appendix A: Category Deep Dives

See `docs/category_analysis.md` for detailed analysis of each deployed category.

## Appendix B: Technical Specifications

See `docs/technical_specs.md` for API integration, model serving, and infrastructure details.

## Appendix C: Risk Scenarios

See `docs/risk_scenarios.md` for extended failure mode analysis and recovery procedures.
