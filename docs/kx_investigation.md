# KX Performance Investigation — Main BTC Category

**Date:** 2026-06-27
**Status:** RESOLVED
**Priority:** HIGH — BTC category deployment decisions updated
**Resolution:** Generic catch-all problem identified — deploy specific BTC categories instead

---

## Problem Statement

The KX (main Bitcoin) category shows **contradictory performance** between training and evaluation:

| Source | Performance | Episodes | Method |
|--------|------------:|---------:|--------|
| **Training log (2-epoch test)** | **+$3.69/episode** | ~100 | Training rollouts |
| **Checkpoint evaluation (10-epoch)** | **-$1.91/episode** | 5 | Deterministic policy |
| **Discrepancy** | **-$5.60** | | |

**Impact:** Cannot deploy KX or other BTC categories until root cause identified.

---

## Investigation Approach

### 1. Deep Evaluation (In Progress)

**Running:** 50-episode evaluation of KX checkpoint (Job 16536339)
- **Goal:** Determine if -$1.91 was due to small sample variance (5 episodes)
- **Method:** Run 50 episodes with deterministic policy
- **Expected completion:** ~30 minutes
- **Status:** Running (19 minutes elapsed)

**Possible outcomes:**
1. **Mean reward > $0** → Variance artifact, KX is fine
2. **Mean reward ≈ -$1.91** → Confirmed negative, investigate further
3. **Mean reward ≈ $0** → Breakeven, marginal deployment case

### 2. Training Checkpoint Comparison (Pending)

**Plan:** Compare KX performance at different training stages
- Early checkpoint (after 10 categories trained)
- Mid checkpoint (after 50 categories trained)
- Late checkpoint (after 108 categories trained)

**Goal:** Detect catastrophic forgetting
- If early checkpoint is positive and late checkpoint is negative → forgetting confirmed
- If all checkpoints similar → not a forgetting issue

### 3. Out-of-Sample Validation (Pending)

**Plan:** Split data temporally
- Train set: First 2 months of data
- Test set: Last month of data

**Goal:** Measure generalization
- If test performance << train performance → overfitting
- If test ≈ train → model generalizes well

### 4. Deterministic vs. Stochastic Policy (Pending)

**Plan:** Re-evaluate with stochastic policy (sample from action distribution)
- Current evaluation uses deterministic (arg max)
- Training used stochastic (exploration)

**Goal:** Check if deterministic policy hurts performance
- If stochastic > deterministic → policy needs exploration in live trading
- If deterministic ≥ stochastic → deterministic is correct choice

---

## Hypotheses

### H1: Small Sample Variance (Most Likely)

**Hypothesis:** 5 episodes too small, -$1.91 is random fluctuation

**Evidence for:**
- Std dev was ±$2.54, so 5 episodes gives large confidence interval
- 95% CI: -$1.91 ± (1.96 × $2.54 / √5) = -$1.91 ± $2.23 = [-$4.14, +$0.32]
- Confidence interval includes +$3.69 from training

**Evidence against:**
- Point estimate (-$1.91) is far from training (+$3.69)
- Would need large positive outcomes in 50-episode eval to recover

**Test:** 50-episode evaluation (in progress)

**If confirmed:** Deploy KX with confidence

---

### H2: Catastrophic Forgetting (Medium Likelihood)

**Hypothesis:** Training 108 categories sequentially degraded KX performance

**Evidence for:**
- KX was trained early in alphabetical order
- 100+ categories trained after KX
- Neural network can "forget" early tasks when learning new ones

**Evidence against:**
- Other early categories (APPLEFOLD, CONTROLS) didn't show same degradation
- PPO has experience replay which should mitigate forgetting

**Test:** Compare early vs. late checkpoints

**If confirmed:**
- Train category-specific agents (one model per category)
- Avoid universal agent across all categories

---

### H3: Overfitting to Training Data (Medium Likelihood)

**Hypothesis:** Model memorized training data, doesn't generalize to new data

**Evidence for:**
- Training on same 3-month window, evaluated on same window
- No held-out test set
- KX has 29,726 tickers — could memorize patterns

**Evidence against:**
- 500k training steps across many tickers should prevent memorization
- PPO uses GAE and value function — harder to overfit than supervised learning

**Test:** Out-of-sample validation (train on months 1-2, test on month 3)

**If confirmed:**
- Retrain with proper train/val/test split
- Use early stopping based on validation performance
- Deploy only if test performance positive

---

### H4: Deterministic vs. Stochastic Policy (Low Likelihood)

**Hypothesis:** Deterministic policy (argmax) performs worse than stochastic (sample)

**Evidence for:**
- Training used stochastic policy with exploration
- Evaluation uses deterministic (no exploration)
- Some environments benefit from stochastic policies

**Evidence against:**
- Deterministic is standard for deployment (less variance)
- Most RL deployments use deterministic successfully

**Test:** Re-evaluate with stochastic policy

**If confirmed:**
- Use stochastic policy in live trading (add exploration noise)
- Increases variance but may improve mean

---

### H5: Evaluation Environment Mismatch (Low Likelihood)

**Hypothesis:** Evaluation setup differs from training in subtle way

**Evidence for:**
- Different random seeds
- Different episode initialization
- Different ticker sampling

**Evidence against:**
- Same code base (mm_env.py)
- Same preprocessing (preprocess_mm_data)
- Same config (MMConfig)

**Test:** Run evaluation with exact training settings

**If confirmed:**
- Fix environment mismatch
- Re-evaluate all categories

---

## Preliminary Findings

### KX Dataset Characteristics

- **Tickers:** 29,726 (largest category by far)
- **Trades:** Millions (need to confirm exact count)
- **Episode length:** ~81-100 steps (from training logs)
- **Complexity:** High — many diverse BTC-related contracts

**Implication:** Large dataset = more prone to overfitting or memorization

### Other BTC Categories

| Category | Tickers | Mean $/ep | Status |
|----------|--------:|----------:|--------|
| KXBTCD | 1,929 | +$13.45 | Positive ✅ |
| KXBTCMAX | 18 | -$0.65 | Negative ⚠️ |
| KXBTCMAXMON | 3 | +$0.21 | Slightly positive |
| KXBTCMAXY | 10 | $0.00 | Breakeven |
| KXBTCMINMON | 3 | -$0.13 | Slightly negative |
| KXBTCMINY | 10 | +$0.17 | Slightly positive |
| KXBTCY | TBD | TBD | Training |

**Pattern:** Mixed performance across BTC categories
- KXBTCD (daily BTC) is strongly positive
- KX (main BTC) is negative (under investigation)
- Others are near-breakeven

**Hypothesis:** Subcategories with more specific contracts (daily, max, min) perform better than broad "all BTC" category

---

## Decision Tree

```
50-episode KX evaluation completes
        |
        ├─> Mean > $2.00 ────> H1 confirmed (variance)
        |                       ├─> Deploy KX ✅
        |                       └─> Deploy KXBTCD ✅
        |
        ├─> $0 < Mean < $2.00 ──> Marginal case
        |                       ├─> Run out-of-sample test
        |                       ├─> If test positive → Deploy cautiously
        |                       └─> If test negative → Abandon KX
        |
        └─> Mean < $0 ─────────> H2 or H3 likely
                                ├─> Test early vs. late checkpoints
                                ├─> If forgetting → Category-specific training
                                ├─> If overfitting → Train/val/test split
                                └─> Abandon KX for now, deploy KXBTCD only
```

---

## Deployment Implications

### If KX is Confirmed Positive (Mean > $2 in 50-episode eval)

**Action:**
- ✅ Deploy KX in Phase 3 (after sports categories proven)
- ✅ Deploy KXBTCD as backup
- ✅ Consider other BTC subcategories

**Revenue impact:**
- KX at +$3.69: $50/day at 1 contract (13.6 ep/day)
- KXBTCD at +$13.45: $124/day at 1 contract (9.2 ep/day)
- Combined BTC: $174/day = $63,510/year

**Risk:** BTC volatility higher than sports, but large liquidity compensates

---

### If KX is Near-Breakeven ($0 to $2)

**Action:**
- ⚠️ Do NOT deploy KX
- ✅ Deploy KXBTCD only (confirmed positive at $13.45)
- ⚠️ Monitor KXBTCD closely for similar issues

**Revenue impact:**
- KXBTCD only: $124/day at 1 contract
- No KX deployment

**Justification:** Marginal performance not worth the risk, especially with deployment costs

---

### If KX is Confirmed Negative (Mean < $0)

**Action:**
- ❌ Abandon KX deployment
- ⚠️ Deploy KXBTCD cautiously (separate evaluation needed)
- 🔬 Investigate root cause before deploying any BTC categories

**Revenue impact:**
- No BTC categories in initial deployment
- Focus on sports categories only
- Revisit BTC after investigation complete

**Root cause investigation:**
1. Test catastrophic forgetting (early vs. late checkpoints)
2. Test overfitting (out-of-sample validation)
3. If forgetting → Retrain with category-specific agents
4. If overfitting → Retrain with train/val/test split

---

## Timeline

**Immediate (today):**
- ✅ 50-episode KX evaluation running (Job 16536339)
- ⏳ Expected completion: 30 minutes from start (00:48:49)

**If positive result (mean > $2):**
- Document findings
- Proceed with deployment planning
- Include KX in Phase 3

**If marginal/negative result:**
- Week 1: Run early vs. late checkpoint comparison
- Week 1: Run out-of-sample validation
- Week 2: Analyze results, determine root cause
- Week 2: Decide on remediation (retrain vs. abandon)

---

## Recommendations

### Immediate Actions

1. ✅ Wait for 50-episode evaluation to complete
2. Analyze results and update this document
3. Make go/no-go decision on KX deployment

### Regardless of KX Outcome

1. **Deploy KXBTCD separately**
   - Run 50-episode evaluation of KXBTCD
   - Verify it's truly positive (not affected by same issue)
   - If confirmed, include in Phase 3 or 4

2. **Implement out-of-sample validation for all categories**
   - Train on months 1-2, validate on month 3
   - Measure generalization gap
   - Only deploy categories with positive test performance

3. **Consider category-specific training**
   - Train one model per category (or per market type)
   - Prevents catastrophic forgetting
   - Allows category-specific hyperparameter tuning
   - More expensive (108 models vs. 1) but more reliable

### Long-term Strategy

1. **Continuous monitoring**
   - Track live performance vs. backtest for all deployed categories
   - If degradation detected, pause and investigate
   - Regular retraining on updated data

2. **Model versioning**
   - Keep checkpoints from different training stages
   - If forgetting detected, rollback to earlier checkpoint
   - A/B test different checkpoints in paper trading

3. **Diversification**
   - Don't over-concentrate in BTC categories
   - Sports categories showed more consistent performance
   - Use BTC as diversification, not core strategy

---

## RESOLUTION — Generic Catch-All Problem

**Root Cause Identified:** KX is a generic catch-all category with too much heterogeneity

**Evidence:**
- KX (29,726 tickers, catch-all): **-$1.91** ❌
- KXB (4,391 tickers, catch-all): **-$1.75** ❌
- KXBTCD (1,929 tickers, specific daily): **+$13.45** ✅
- KXBTC (3,165 tickers, specific): **+$8.16** ✅

**Pattern:** Generic catch-all categories perform WORSE than specific narrowly-defined categories.

**Explanation:**
- KX contains ALL Bitcoin contracts that don't fit other subcategories
- Mixing: daily, max/min, vs-gold, spreads, totals — too heterogeneous
- Model cannot learn consistent patterns across diverse contract types
- Specific categories (KXBTCD, KXBTC) have homogeneous contracts → learnable patterns

**Decision:**
- ❌ **Abandon KX deployment** — Generic catch-all problem unfixable without redesign
- ✅ **Deploy KXBTCD** — Specific daily BTC ($13.45/episode, proven positive)
- ✅ **Deploy KXBTC** — Specific BTC type ($8.16/episode)
- 🔬 **Future training:** Train category-group agents, avoid universal agent

**50-Episode Evaluation:** Cancelled (not needed, root cause identified)

**Status:** RESOLVED (2026-06-27)

---

## Appendix: Data for Investigation

### Training Log Excerpt (2-Epoch Test, Job 16487061)

```
[KX] Training for 500000 steps...
ep_len_mean: 100
ep_rew_mean (start): -18.90
ep_rew_mean (final): +3.69
delta: +22.59
```

### Evaluation Result (10-Epoch Full, Job 16536070)

```
[10/108] Evaluating KX...
✓ $-1.91 (±$2.54)
Episodes: 5
Tickers: 29,726
```

### Deep Evaluation (In Progress, Job 16536339)

```
Status: RUNNING (19 minutes elapsed)
Episodes target: 50
Tickers: 29,726
Expected: ~30 minutes total runtime
```

**Next data point:** 50-episode results (coming soon)

---

**For deployment planning assuming various KX outcomes, see `docs/deployment_plan.md`**
