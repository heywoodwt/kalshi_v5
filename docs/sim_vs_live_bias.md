# Sim-vs-Live Bias in the Market-Making Simulator

**Document date:** 2026-07-01  
**Purpose:** Technical explanation of the adverse-selection bias in `rl_bot/mm_env.py` and the calibration method to measure it.

---

## Executive Summary

The training/eval simulator (`rl_bot/mm_env.py`) computes market-making PnL in a way that **ignores adverse selection** — the cost of being filled precisely when the counterparty is right and you are about to be wrong. The result is an upward-biased PnL estimate: +9,744 in eval, but −$0.31 in live (on KXBTC, 200 most recent fills).

This is not noise, bad luck, or a tuning problem. It is a **structural flaw in the label-generating process** that cannot be fixed by retraining or hyperparameter search. The simulator's objective function is anti-correlated with live profitability in regimes where adverse selection is large (high-vol categories like KXBTC). Only fixing the simulator fixes the model.

This document explains:
1. **What the bias is** — formal definition and magnitude
2. **Why it exists** — the mechanism in the code
3. **How to measure it** — markout analysis via `analyze_sim_vs_live.py`
4. **What to do about it** — ranked by impact

---

## 0. Statistical Framework (for Data Scientists)

### Problem Formulation

We are training a policy π to maximize expected PnL under a simulator that generates labels (rewards) via a biased estimator. Formally:

Let:
- **π(a | s)** = agent policy: action distribution given state
- **s** = observation (orderbook, inventory, mid price, etc.)
- **a** = action (half_spread, skew)
- **r_sim** = simulator reward (what we optimize)
- **r_live** = true reward (what we want)

The simulator computes:

```
r_sim = Δ(realized_PnL_sim + unrealized_PnL_sim)
      = spread_capture - inventory_penalty
```

Where `spread_capture` is marked against the contemporaneous VWAP (pre-fill mid). The true reward should be:

```
r_live = E[Δ(realized_PnL_live + unrealized_PnL_live) | filled]
       = spread_capture - adverse_selection - (taker_fee - maker_fee) - slippage
```

The difference is:

```
bias = E[r_sim - r_live | action, filled]
     = -E[adverse_selection | filled]
```

### Why This Is Selection Bias

**Definition:** Selection bias occurs when the sample used to estimate a parameter (or label) is not representative of the population because of systematic non-random selection.

In our case:
- **Population:** All possible market states and order flows.
- **Sample:** Historical trades that crossed our quotes (selected by the fill event).
- **Parameter of interest:** True edge = spread - adverse selection.
- **Biased estimator:** Simulator's edge = spread (conditional on fill, but marked as unconditional).

**Formally**, let I_t = 1 if filled at time t, 0 otherwise. The simulator computes:

```
PnL_sim = Σ_t I_t × [spread_t - 0]
        = E[PnL | I=1] (simulator's view)

PnL_live = Σ_t I_t × [spread_t - adverse_t]
         = E[PnL | I=1] (true value)
```

So: `bias_per_fill = E[adverse_t | I_t=1]`

The simulator sets `E[adverse_t | I_t=1] = 0` (ignores it), when in reality it's positive (you get filled when the market is moving against you).

### Connection to Causal Inference

This is a **collider bias** problem in causal DAG terms:

```
         Informed_Flow
              |
              ↓
    Price_Movement → Fill_Event ← Agent_Quote_Quality
```

The fill event is a **collider**: conditioning on it (which the simulation does) opens a path between informed flow and your quote quality. In other words, fills that occur are enriched for informed flow, even if your quotes are mediocre. The simulator treats fills as random samples; they're actually biased samples.

---

## 1. The Statistical Flaw: Selection Bias in Labels

### The Problem

In real market making, your fills are a **non-random sample** of order flow. You get filled when someone *wants* to trade against you — and they disproportionately want to when they're informed (right about the next price move).

**Formally:** The simulator samples fills conditioned on the historical price crossing your quote, then evaluates PnL using a statistic that is not independent of that conditioning event. This induces selection bias in the label (PnL).

### The Mechanism

`rl_bot/mm_env.py:568–573` checks if historical trade prints cross your quotes:

```python
buy_through  = (sides == 0) & (prices < bid)      # trade printed below our bid
sell_through = (sides == 1) & (prices > ask)      # trade printed above our ask
```

When a print crosses, the fill happens at `prices[i]` (the print price). The simulator then marks your inventory at the **window's VWAP** (`mm_env.py:499–507`):

```python
vwap = window.get("vwap")
if vwap is not None and vwap > 0:
    self._mid = vwap
```

So immediately after you "buy at the bid," you are marked at mid and book roughly **+half_spread** of unrealized profit (`mm_env.py:405–409`):

```python
def _unrealized_pnl(self) -> float:
    if self._inventory == 0:
        return 0.0
    return self._inventory * (self._mid - self._avg_entry_price)
```

### Why This Is Wrong

The simulator:
- **Fills you** based on historical price movement (selection event).
- **Marks you** using a statistic (VWAP) that already contains the very trade that filled you.
- **Never lets the price move against your new position** — the VWAP is fixed for the window.

In reality, the trade that filled you is a signal. The counterparty who crossed your quote was *more likely to be right*. After the fill, the true mid drifts against your new inventory with probability > 0.5. The simulator skips this entirely.

### The Bias Magnitude

The simulator credits `spread ≈ half_spread × 2` per fill. The true edge (spread minus adverse-selection cost) is roughly:

```
true_edge = spread - E[price_drift | filled]
```

In high-vol categories (KXBTC, vol ≈ 0.070), the drift is large: adverse selection ≈ half the spread. In low-vol categories (ADP, vol ≈ 0.021), it's small.

**Live data confirms:**
- KXBTC: eval = +9,744 PnL, live = −$0.31 (94.5% of fills are taker, so paying spread not earning it)
- KXADP/gas: modest live profitability (low vol, 1-directional flow, low informed-trader density)

The gap: sim minus live ≈ **$9,700 per backtest**, concentrated in high-vol markets.

---

## 2. Why This Happens: Root Causes

### Root Cause 1: Fill Selection (Non-Causality)

The trades in your training data occurred in a world **where your orders did not exist**. Replaying them as fills assumes:
- Your liquidity changes nothing — no queue effects, no counterparty redirect, no market impact.
- Every trade at your quote level fills you immediately.

Reality: Resting orders sit in a FIFO queue. Early queue position (priority) is valuable precisely when everyone else wants to trade the same direction. That correlation with one-directional informed flow means your queue position *signals* you will be picked off.

The `count // (queue_competitors + 1)` haircut (`mm_env.py:590`) shrinks fill *size* but leaves the **selection bias untouched**: you still get filled on the trades where informed flow is strongest, and the mid still drifts against you.

### Root Cause 2: Mark-to-Mid Assumption

The simulator assumes you can exit at mid. In live trading:
- **The bot mostly pays the spread**, not earns it: 94.5% taker fills in recent data.
- Exit orders are **stop-loss and expiry IOCs** that cross the spread (`live_trader_v2.py:1181–1202`), not resting maker bids.
- This reverses the sign of the spread term from **+** (what the sim credits) to **−** (what you pay).

### Root Cause 3: No Adverse-Selection Modeling

The simulator's fill model is *unconditional* on the informed/uninformed status of the flow. It samples fills uniformly from historical prints that cross your quote. A better model would:
- Mark inventory at the **post-fill price path**, not the pre-fill VWAP.
- Adjust fill *probability* based on how much volume traded *through* your level after you'd join the queue.
- Use a simple heuristic: vol that "through-fills" should be rare; vol at-touch should depend on queue depth.

Currently, the first term (spread capture) is accurate, but the second term (E[drift | filled]) is set to zero. The simulator optimizes the policy to maximize `spread − 0`, which is equivalent to maximizing `adverse selection`. Live markets select *against* that policy.

---

## 3. How to Measure It: Markout Analysis (Statistical Method)

### Statistical Formulation

**Objective:** Estimate the conditional bias term `E[adverse_t | I_t = 1]` using realized markouts.

**Estimand:** For a fill i at timestamp t_i with price p_i, we want to estimate:

```
adverse_i(Δt) = E[price(t_i) | history up to t_i] - E[price(t_i + Δt) | history up to t_i]
               ≈ mid(t_i^-) - mid(t_i + Δt)
```

Where:
- `mid(t_i^-)` = mid price in the instant before the fill (strictly before, to avoid look-ahead bias)
- `mid(t_i + Δt)` = mid price at Δt seconds after the fill

The key insight is that in an efficient market without your order, `E[mid(t) | history] = mid(t^-)` (martingale). But after the fill, new information arrives (the fill itself is a signal), so the mid drifts. The drift captures adverse selection.

**Estimator:** We use the **last trade price** as a noisy proxy for the true mid:

```
mid(t)_hat = price of last trade at or before time t
```

This gives the sample markout:

```
markout_i(Δt) = signed_i × (mid_hat(t_i + Δt) - p_i)
adverse_i(Δt) = entry_edge_i - markout_i(Δt)
              = signed_i × (mid_hat(t_i^-) - mid_hat(t_i + Δt))
```

**Aggregation:** Across n_cat fills in a category at horizon Δt:

```
adverse_avg(Δt) = (1/n_cat) Σ_i adverse_i(Δt)

bias_total(Δt) = Σ_i adverse_i(Δt) × count_i

Where count_i = number of contracts in fill i
```

The aggregation weights by contract count because a 50-contract fill that drifts 1 cent is 50× worse than a 1-contract fill with the same drift.

### Why Last Trade Price Is Valid

**Assumption:** The last trade price is a sufficient statistic for the mid, conditional on the information set at time t.

**Justification:**
1. **Informational efficiency:** Under market-microstructure models (Kyle, 1985; Glosten-Milgrom, 1985), the last trade price is the posterior expected value of the contract, given the trade flow observed so far.
2. **Bid-ask symmetry:** Kalshi markets are binary (YES/NO are complements), so the bid and ask are symmetric around 0.50, and the mid is the average. The last trade price is close to this.
3. **Consistency with simulator:** The simulator itself marks using VWAP (a time-weighted average of mid prices from trade prints). Using the last trade price is thus *more conservative* than VWAP (it ignores the effect of large intermediate trades), so the bias we measure is a lower bound.

### Hypothesis Test: Is Adverse Selection Significant?

**Null hypothesis (H0):** `E[adverse | filled] = 0` (the simulator is unbiased).  
**Alternative (H1):** `E[adverse | filled] > 0` (adverse selection is real).

**Test statistic:** 
```
t = mean(adverse_i) / (std(adverse_i) / sqrt(n))
```

Under H0, this follows approximately a t-distribution with n-1 degrees of freedom.

**Rejection rule:** If t > t_{0.05, n-1}, reject H0 at 5% significance level.

**Power analysis:** With typical adverse selection of 1-2 cents per contract and std of 2-3 cents, you need ~100 fills per category to achieve 80% power.

### Robustness: Sensitivity to Mid Estimation

The main source of noise is the last-trade-price estimate of the true mid. To validate:

1. **Compare to bid-ask midpoint:** When orderbook data is available, compute the true spread mid and see if results are qualitatively similar.
2. **Horizon sensitivity:** Plot `adverse(Δt)` for Δt = 10s, 30s, 60s, 300s, 3600s. If adverse selection is real, it should be:
   - Large at short horizons (10-60s): immediate feedback from the informed counterparty.
   - Smaller at long horizons (3600s): price mean-reverts, reducing the drift component.
   
   If adverse selection is spurious (e.g., random noise in the mid), it should be independent of Δt.

3. **Per-category breakdown:** If adverse selection is a simulator artifact (not real market microstructure), it should be uniform across categories. Instead, expect high-vol categories to show 5-10× larger adverse selection than low-vol.

### Example: KXBTC vs KXADP

**KXBTC (high-vol, vol ≈ 0.070):**
- Typical fill: 1 contract at price 0.50.
- Mid before fill: 0.505 (we quoting at bid 0.495).
- Mid 30s after: 0.515 (market moved up by 1 cent).
- Entry edge: 0.505 - 0.50 = 0.5 cents (half-spread).
- Markout(30s): 0.515 - 0.50 = 1.5 cents (we're now long and the market moved against us).
- Adverse: 0.5 - 1.5 = -1.0 cents (lost 1 cent to adverse selection).

Aggregated over 100 such fills: bias_total = -100 cents = -$1.00 on 100 contracts.

**KXADP (low-vol, vol ≈ 0.021):**
- Typical fill: 1 contract at price 0.07 (very low price, thin market).
- Mid before: 0.072.
- Mid 30s after: 0.071 (market barely moved, only -1 millicent).
- Entry edge: 0.072 - 0.07 = 0.2 cents.
- Markout(30s): 0.071 - 0.07 = 0.1 cents.
- Adverse: 0.2 - 0.1 = 0.1 cents (lost only 0.1 cents).

Aggregated over 100 fills: bias_total = -$0.10 on 100 contracts.

**Gap:** KXBTC bias is 10× larger than KXADP, which explains why sim PnL is so much higher than live PnL in KXBTC and near live PnL in KXADP.

---

### The Method

For each real fill, compute:

```
entry_edge   = signed × (mid_before_fill − fill_price)
markout(dt)  = signed × (mid_at_fill_ts+dt − fill_price)
adverse(dt)  = entry_edge − markout(dt)
             = signed × (mid_before_fill − mid_at_fill_ts+dt)
```

Where:
- `signed` = +1 if the fill lengthens YES inventory, −1 if shortens.
- `mid_before_fill` = last trade price strictly before the fill (binary search on ts).
- `mid_at_fill_ts+dt` = last trade price at or before fill_ts + dt seconds.
- `entry_edge` = the spread the simulator credits you.
- `markout(dt)` = the edge that survives after dt seconds (what a real trader could exit at).
- `adverse(dt)` = the shortfall — adverse selection paid.

### Why This Works

The method **measures the same quantity the simulator ignores**: the price drift after a fill. By comparing `entry_edge` (what sim pays you) to `markout(dt)` (what you can actually exit for), we quantify the bias.

Key properties:
- **Fair:** Uses the same mid-price basis (last trade) that the simulator marks against.
- **Direct:** Adverse selection is just the difference in mid between before and after the fill.
- **Per-category:** Aggregated by category, it shows where the bias is largest (high-vol) and smallest (low-vol).

### Implementation: `analyze_sim_vs_live.py`

Located at `/rl_bot/analyze_sim_vs_live.py`, this script:

1. **Fetches fills** from the Kalshi API or a cached parquet.
2. **Builds a mid-price index** from the trades parquet (ticker → time-sorted array of (ts, price)).
3. **For each fill**, binary-searches the mid just before and at dt seconds after.
4. **Computes** entry_edge, markout(dt), and adverse(dt).
5. **Aggregates per (category, horizon)** and outputs a CSV.

**Usage:**

```bash
# Fetch 48 hours of fills from the API:
python analyze_sim_vs_live.py --hours 48 --trades output/rl_kalshi_trades_3mo.parquet

# Or use cached fills:
python analyze_sim_vs_live.py --fills-parquet output/live_fills.parquet --trades output/rl_kalshi_trades_3mo.parquet
```

**Output:** `output/sim_vs_live_calibration.csv` with columns:
- `category`: market category (e.g., KXBTC, KXADP).
- `horizon_s`: time horizon (30, 60, 300 seconds).
- `n_fills`: number of fills matched to trade data.
- `sim_edge_per_contract`: simulator's credited edge (−0.5 = paying half a cent per contract).
- `live_edge_per_contract`: realized edge after drift (−0.52 = only slightly worse if drift is small).
- `adverse_per_contract`: the gap (adverse selection paid).
- `bias_total`: total adverse selection cost over all fills (dollars).
- `net_live_total`: live edge minus maker fees.

**Interpretation:**
- **`adverse_per_contract` > 0** means the market drifts against your fills (informed flow hitting you).
- **`adverse_per_contract` ≈ 0** means the price is stable (uninformed or balanced flow).
- **`bias_total` > sim_edge_total / 2** suggests adverse selection is eating more than half the spread you thought you'd capture.

---

## 3.5. Parametric Model of Adverse Selection (Optional Deeper Analysis)

For a data scientist who wants to move beyond descriptive markouts, you can fit a parametric model of adverse selection as a function of observable market features. This is useful for:
- **Out-of-sample prediction:** Forecast adverse selection for fills not yet observed.
- **Causal modeling:** Separate genuine adverse selection from confounding factors (e.g., liquidity shocks).
- **Policy learning:** Understand how market conditions (volatility, volume, book imbalance) predict the cost of a fill.

### Simple Linear Model

Fit a regression:

```
adverse_i = β_0 + β_vol × volatility_i + β_imbal × imbalance_i + β_depth × depth_i + ε_i
```

Where:
- `volatility_i` = realized volatility in the 60s before the fill (rolling stddev of mid).
- `imbalance_i` = order flow imbalance (buy volume - sell volume) in the 60s before the fill.
- `depth_i` = top-of-book depth (sum of bid and ask depth at L0) just before the fill.

**Interpretation:**
- `β_vol > 0` means high-vol markets have more adverse selection (informed traders are more confident when prices are volatile).
- `β_imbal > 0` (if positive) means positive order flow imbalance predicts adverse selection against you (your quote is on the wrong side of the flow).
- `β_depth < 0` means deep books have less adverse selection (more competition, less informed flow per fill).

### Hierarchical Model (Category-Level Random Effects)

If you have many categories, fit a mixed-effects model:

```
adverse_i,cat = β_0 + b_cat + β_vol × volatility_i + ε_i

where b_cat ~ N(0, σ_b²)
```

This partitions the adverse selection into:
- **Fixed effect (β_0):** Global average adverse selection across all categories.
- **Random intercept (b_cat):** Category-specific adjustment (e.g., KXBTC has +2 cents/contract extra adverse selection vs. the global mean).
- **Coefficient (β_vol):** Shared effect of volatility across all categories.

**Usefulness:** You can then estimate category-specific bias adjustments:

```
predicted_adverse_KXBTC = β_0 + b_KXBTC + β_vol × typical_volatility_KXBTC
```

And adjust your risk budgets or deployment decisions per category.

### Validation: Cross-Validation and Out-of-Sample R²

1. **Split data:** 80% training, 20% test.
2. **Fit model on training data.**
3. **Predict adverse selection on test data.**
4. **Compute test R²:**

```
R²_test = 1 - (SS_residual / SS_total)
```

If R² > 0.5, the model is predictive; if R² ≈ 0.1, volatility/imbalance/depth don't explain adverse selection well (suggesting it's mainly white noise, not systematic).

---

## 3.6. Sample Size and Statistical Power (Planning Your Validation Run)

Before running the calibration, determine how many fills you need to detect a real adverse-selection effect with high confidence.

### Power Calculation

**Scenario:** You want to detect a mean adverse selection of μ_true = 0.005 (5 millicents per contract) with power = 0.80 (80% chance of detecting it if it's real).

**Assumptions:**
- Null hypothesis: μ = 0 (no adverse selection).
- Alternative hypothesis: μ = 0.005.
- Significance level: α = 0.05 (two-tailed test).
- Estimated standard deviation: σ ≈ 0.003 (3 millicents, typical of thin markets).

**Cohen's d effect size:**
```
d = μ / σ = 0.005 / 0.003 ≈ 1.67
```

**Sample size (from standard power tables):**
For d = 1.67, α = 0.05, power = 0.80:
```
n ≈ 14 fills
```

For d = 0.5 (weaker effect, harder to detect):
```
n ≈ 64 fills
```

For d = 0.2 (very weak, practically undetectable):
```
n ≈ 400 fills
```

### Practical Guidance

| Effect Size | Adverse Selection | Fills Needed (80% power) | Real-Time Hours* |
|-------------|-------------------|-----------------------|------------------|
| Large (d=1.0) | 3-5 cents | ~26 | ~1 week |
| Medium (d=0.5) | 1-2 cents | ~64 | ~2-3 weeks |
| Small (d=0.2) | 0.3-0.5 cents | ~400 | ~2 months |

*Assuming 10-20 fills per day during active trading.

### Minimum Viable Validation

**To prove the bias is real:** Collect at least **100 fills per category** at a 60-second horizon, then run the t-test. This gives 80% power to detect adverse selection ≥ 0.5 cents per contract.

**To validate per-category:** KXBTC, KXADP, KXAAAGASM = 300 fills total. With 10-20 fills/day, that's 2-4 weeks.

**To validate per-horizon:** Add 30s, 300s, 3600s = 400 fills × 4 horizons = 1,600 fills. This takes 2-3 months but gives a complete picture of how adverse selection evolves over time.

### Stratification (Increase Power Without More Data)

If fills are heterogeneous (some are 1 contract, some are 50), stratify by contract size:

```
For each (category, contract_size_bucket):
  Compute mean adverse_i
  Test within each bucket
  Combine p-values across buckets using Fisher's method
```

This can 2-3× your power because you're reducing within-group variance.

---

## 4. What to Do: Ranked by Impact

### 1. **Validate the Simulator Against Real Fills (HIGHEST IMPACT)**

Before retraining, run the calibration script on recent live fills:

```bash
python analyze_sim_vs_live.py --hours 168 \
  --trades output/rl_kalshi_trades_3mo.parquet \
  --out output/calibration_latest.csv
```

**Action:** For each category:
- If `adverse_per_contract` at 60s is > 1 cent / 100 contracts ≈ 0.0001, you have a real adverse-selection problem.
- If it's ≈ 0, the simulator is accurate for that category and retraining *may* help.

**Why first:** If the bias is real and large, all the retraining in the world won't fix it. Fixing the simulator is a prerequisite.

### 2. **Fix the Simulator's Fill Model**

After the fill, mark inventory at the **post-fill price** (or a moving average), not the pre-fill VWAP. This single change reintroduces the adverse-selection cost:

```python
# Current (biased):
def step(self, action):
    ...
    self._mid = vwap  # mid for this window (contains the fill!)
    ...
    new_value = self._realized_pnl + self._unrealized_pnl()  # marked at vwap

# Fixed:
def step(self, action):
    ...
    self._mid = vwap  # mid from *earlier* in the window (before any new fill)
    ...
    if fill_happened:
        # Mark the new inventory at the *post-fill* mid
        post_fill_mid = next_window_vwap  # or mid after dt seconds
        unrealized_drift = self._inventory * (self._mid - post_fill_mid)
        ...
```

This makes the simulator pessimistic about fills (which is realistic) and teaches the policy to be conservative.

### 3. **Add Realized Volatility as an Observation Feature**

Give the model a 20th dimension: rolling stddev of mid price over the last N windows. This lets the policy learn to widen spreads in volatile regimes instead of needing separate models per volatility bin:

```python
# In _build_obs():
volatility = np.std(self._mid_history[-20:]) if len(self._mid_history) >= 20 else 0.01
obs[19] = np.clip(volatility / 0.05, 0.0, 1.0)  # normalized
```

### 4. **Filter Deployments by Category Volatility**

In `live_trader_v2.py`, compute rolling within-ticker price volatility and skip quoting when vol > 0.05 (the threshold dividing MM-friendly from MM-hostile markets):

```python
# In _build_observation():
if len(self.state.mid_history[ticker]) >= 10:
    vol = np.std(self.state.mid_history[ticker][-10:])
    if vol > 0.05:
        return None  # skip quoting in this market
```

### 5. **Model Taker-Fill Fraction in Simulation**

Live data shows 94.5% taker fills. The simulator assumes 100% maker. Adjust the fill cost model to charge taker fees on a fraction of fills:

```python
# In _fill_buy():
if random.random() < 0.945:  # taker fill (live observation)
    fee = compute_taker_fee(size, price, 0.0175)
else:
    fee = compute_maker_fee(size, price, 0.0175)
```

This makes the simulator pessimistic in the direction of reality and should improve eval-to-live correlation.

### 6. **Reduce Through-Fill Generosity**

Current: `through_fill_haircut = 0.50` (assume you get 50% of through volume).  
Better: `through_fill_haircut = 0.33` (assume you get 1/3, more realistic queue position).

```python
fill_count = int(counts[j] * 0.33)  # down from 0.50
```

### 7. **Retrain per-Volatility-Regime Models**

Once the simulator is fixed, train two models:
- **Low-vol model** (KXAAAGASM, KXADP, KXBTCD): trained on vol < 0.05 markets.
- **High-vol model** (KXBTC, KXMLBGAME): trained on vol >= 0.05 markets.

Deploy the appropriate model per category. The high-vol model may learn to quote very wide (or sit out), which is the correct behavior.

### 8. **Sync with Kalshi S3 for 3-Month Orderbook Data**

The current orderbook data (831 snapshots) is too sparse. Collect from the `kalshi-data-prod` bucket (us-east-2) on S3 for 3 months targeting the top 20 categories. This gives training realistic book depth and imbalance signals.

---

## 4.9. Why This Cannot Be Fixed By Standard ML Techniques (For Data Scientists)

If you have an MS in data science, you might ask: "Can't we just:
- Use domain randomization to make the simulator harder?"
- Use adversarial training to make the policy robust?"
- Use different loss functions (e.g., CVaR instead of expectation)?"
- Regularize the policy to be more conservative?"

**Answer: No, none of these fix the underlying bias.** Here's why, in statistical terms:

### The Fundamental Issue: Measurement Error in the Label

This is not a **model bias** (where a simpler model underfits). It's a **label bias** (where the measurement process itself is biased).

**Statistical principle:** No learning algorithm can overcome a systematic bias in the target variable.

Formally, if your training labels are:
```
y_sim = y_true + bias × indicator(selected)
```

Where `bias > 0` and `indicator(selected) = 1` for any fill that occurs, then the regression:
```
y_hat = E[y_sim | X]
```

has expected error:
```
E[error] = E[y_hat - y_true | X]
         = E[bias × indicator(selected) | X]
         > 0
```

No amount of regularization, ensembling, or adversarial training removes the bias term. It's in the data.

### Why Domain Randomization Doesn't Help

Domain randomization (adding noise to spread, depth, volume) makes the simulator noisier but doesn't remove the systematic bias. You're adding variance to an already biased estimator, which *increases* the mean squared error.

**Domain randomization changes:**
```
y_train = y_true + bias + noise
```

The policy still learns `bias` as part of the signal. The policy adapts to noise, but the bias persists.

### Why Regularization Doesn't Help

Regularization (L1, L2, entropy penalty) reduces overfitting to noise. But the bias is not noise; it's a systematic feature of the label-generating process.

**Example:** If the true optimal spread is 2 cents but the simulator credits you 5 cents (due to ignored adverse selection), regularization will keep you at 2 cents (correct) *only if* the regularization is strong enough to outweigh the biased signal (the +3 cent incentive to widen).

In practice, the policy will learn a compromise: quote wider than optimal (say 3 cents) because the simulator kept rewarding that.

### Why Robust Optimization Doesn't Help

Robust optimization (training against worst-case perturbations) assumes the uncertainty is in the environment, not the objective. Here, the uncertainty is in the label.

If you train a policy to be robust to "the market being worst-case adversarial," you'll get a conservative policy that doesn't exploit any edge. That's correct for high-vol markets (where adverse selection is large), but you're doing this globally, not just in high-vol categories.

### The Solution: Unbias the Label

The only way to fix this is to change the label:

```
# Before (biased)
y_sim = spread - 0 = spread

# After (unbiased)
y_live = spread - adverse_selection - fees
```

Once the label is unbiased, standard supervised learning (PPO, DQN, etc.) will work fine. The policy will learn to quote narrowly in high-vol markets (where adverse selection is large) and widely in low-vol markets (where it's small).

**This is why fixing the simulator (step 2 in the roadmap) is prerequisite to any other improvement.**

---

## 4.10. Connection to Omitted Variable Bias (Econometric Perspective)

For a data scientist familiar with econometrics, this problem is a textbook case of **omitted variable bias**.

### The Regression Analogy

Think of the simulator as fitting a regression:

```
PnL_sim = β_0 + β_spread × spread + ε
```

The true model is:

```
PnL_live = β_0 + β_spread × spread + β_adverse × adverse_selection + u
```

If you omit the `adverse_selection` term:

```
β_spread_biased = β_spread_true + bias
```

Where the bias depends on the correlation between `spread` and `adverse_selection`:

```
bias = β_adverse × Cov(spread, adverse_selection) / Var(spread)
```

In practice:
- **Cov(spread, adverse_selection) > 0:** Tight spreads (good quotes) have low adverse selection; wide spreads (bad quotes) have high adverse selection. This is positive correlation.
- **β_adverse < 0:** Adverse selection *reduces* PnL.
- **Result:** β_spread_biased < β_spread_true (the spread term is biased downward).

But wait — the simulator shows *high* spread values. How?

**Answer:** The simulator doesn't use the regression at all. It just computes `PnL_sim = spread` directly (no other features). It's not estimating a regression coefficient; it's using the raw spread as the reward.

So the equivalence is:

```
PnL_sim (what we train on)  = spread
PnL_live (what we care about) = spread - adverse_selection - fees

Bias per fill = PnL_sim - PnL_live = adverse_selection + fees
```

### Correction Methods from Econometrics

If you had the `adverse_selection` variable measured directly (via our markout analysis), you could:

1. **Include it in the reward:** `r_corrected = spread - adverse_selection - fees`
2. **Use it as a control variable:** `r_corrected = spread - Ê[adverse_selection | market_features]`
3. **Instrumental variables:** If adverse selection is measured with error, instrument it with exogenous market conditions (volatility, volume, etc.).

We do step 1: measure adverse selection via markouts, then use it to correct the simulator. Steps 2-3 are more advanced but could be done if you have sparse fill data (need to predict adverse selection from market features, not compute it directly).

### Variance-Bias Tradeoff

**Measuring adverse selection adds variance:** Each fill's markout is noisy (last trade price is a noisy proxy for true mid).

**Omitting it creates bias:** The simulator credits non-existent profits.

Standard stats principle: **Bias beats variance in large samples.** With 100+ fills, the variance in measuring adverse selection is small, and the bias from omitting it is large. Measuring it is the right call.

---

## 5. Empirical Evidence

### From Project Notes (as of 2026-07-01)

**Eval vs live gap by category:**

| Category | Eval PnL | Live PnL | Eval Win% | Live Win% | Notes |
|----------|----------|----------|-----------|-----------|-------|
| KXBTC | +9,744 | −$0.31 | 53.9% | 40% | High vol (0.070), 94.5% taker fills |
| KXADP | n/a | Real profit | n/a | n/a | Low vol (0.021), 94-100% 1-way flow |
| KXAAAGASM | n/a | Real profit | n/a | n/a | Low vol (0.011), uninformed flow |

**Interpretation:**
- KXBTC: Large gap consistent with high adverse selection + taker fills paying spread.
- KXADP/gas: Smaller gap consistent with low vol and uninformed (retail) flow.
- Conclusion: Adverse selection is the dominant term. The simulator is only trustworthy when vol is low *or* flow is uninformed.

### From Markout Analysis (Expected)

When you run `analyze_sim_vs_live.py` on KXBTC fills:
- **30s markout:** `adverse_per_contract` ≈ 0.002–0.005 (2-5 cents lost to drift per contract).
- **60s markout:** `adverse_per_contract` ≈ 0.003–0.006.
- **300s markout:** `adverse_per_contract` ≈ 0.004–0.008.

Multiplied by typical fill counts (1–50 contracts per fill), this explains 1-5 cents of loss per fill — exactly the scale of the eval-vs-live discrepancy.

For KXADP (low vol):
- **30s markout:** `adverse_per_contract` ≈ 0.0001–0.0003.
- Eval-to-live correlation should be much tighter.

---

## 6. Implementation Roadmap

### Phase 1: Measure (this week)

1. Run `analyze_sim_vs_live.py` on 168 hours of recent fills (if live trading is active).
2. Compute `adverse_per_contract` per category.
3. Verify that high-vol categories (KXBTC, KXMLBGAME) show large adverse selection and low-vol categories (KXADP, gas) show small.

**Deliverable:** Calibration CSV showing bias per category.

### Phase 2: Fix Simulator (1–2 weeks)

1. Modify `mm_env.py`: mark positions at post-fill mid, not pre-fill VWAP.
2. Add realized-volatility feature (dimension 20).
3. Retrain on corrected simulator with 3-month trades data.
4. Compare new eval PnL to live on held-out data.

**Deliverable:** Updated `mm_env.py`, new trained model, calibration comparison.

### Phase 3: Deployment (2–3 weeks)

1. Add volatility filter to `live_trader_v2.py` (skip quoting when vol > 0.05).
2. Retrain per-volatility-regime models.
3. Deploy low-vol model on proven categories (ADP, gas).
4. Monitor live PnL vs. new eval PnL for 1 week.

**Deliverable:** Live deployment on 3–5 low-vol categories with eval-to-live correlation > 0.7.

---

## 7. Technical Notes for Implementation

### Fill Data Format (Kalshi API)

```python
fill = {
    "market_ticker": "KXBTC-26JUL01-T50250",
    "action": "buy" or "sell",           # agent action (buy/sell)
    "side": "yes" or "no",               # contract side
    "yes_price_dollars": 0.45,           # price paid (if side=="yes")
    "no_price_dollars": 0.55,            # price paid (if side=="no")
    "count_fp": 5.0,                     # count (float)
    "created_time": "2026-06-30T12:34:56.789Z",
    "ts": 1782772493000,                 # milliseconds
}
```

**YES-equivalent price:**
```python
if fill["side"] == "yes":
    price_yes = fill["yes_price_dollars"]
else:
    price_yes = 1.0 - fill["no_price_dollars"]
```

**Direction (signed):**
```python
signed = 1 if (action == "buy") == (side == "yes") else -1
```

### Trades Data Format

```python
# output/rl_kalshi_trades_june.parquet
trades_df = {
    "ticker": "KXBTC-26JUL01-T50250",
    "yes_price": 0.45,          # yes-side price (same as fill's "price_yes")
    "count": 10,                # volume
    "taker_side": "yes",        # who initiated (not needed for markout)
    "ts": 1782772493000,        # milliseconds
}
```

### Maker Fee Calculation

```python
from rl_bot.reward import compute_maker_fee

fee = compute_maker_fee(contracts=1, price=0.45, fee_rate=0.0175)
# fee ≈ 0.0175 × 1 × 0.45 × (1 - 0.45) ≈ 0.43 cents
```

---

## 8. References

- **Simulator:** `rl_bot/mm_env.py`, lines 405–409, 568–573, 612–614.
- **Live trader:** `rl_bot/live_trader_v2.py`, lines 1008–1060 (fill handling), 1181–1202 (exit logic).
- **Calibration script:** `analyze_sim_vs_live.py`, full markout analysis.
- **Kalshi API:** `rl_bot/kalshi_api.py`, `get_fills()`, `get_orderbook()`.

---

## Conclusion

The sim-vs-live gap is a **label-generation bias**, not a model or data problem. The simulator credits you the spread but hides the adverse-selection cost. Fixing the simulator is prerequisite to any strategy that relies on simulation.

The calibration script (`analyze_sim_vs_live.py`) measures this bias directly from live fills and shows which categories are affected most. Start there: verify the bias is real and quantify it per category. Then fix the simulator, retrain, and redeploy on the low-vol categories where the strategy actually works.
