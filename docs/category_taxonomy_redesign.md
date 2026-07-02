# Category Taxonomy Redesign — Future Training Strategy

**Date:** 2026-06-27
**Purpose:** Improve model performance by avoiding generic catch-all categories
**Problem:** Universal agent trained on 2,327 heterogeneous categories performs poorly on generic catch-alls
**Solution:** Group-specific agents with homogeneous category clusters

---

## Problem Analysis

### Current Approach (Ineffective)

**Single universal agent** trained on ALL 2,327 categories:
- Generic catch-alls (KX, KXB) perform poorly
- Catastrophic forgetting as new categories added
- No specialization for category types
- One-size-fits-all fails for diverse markets

**Results:**
- KX (29,726 tickers): **-$1.91** ❌ (too heterogeneous)
- KXB (4,391 tickers): **-$1.75** ❌ (too heterogeneous)
- Specific categories: **Much better** (KXBTCD +$13.45 ✅)

### Root Cause: Heterogeneity

**Generic catch-all problem:**
- KX contains: daily BTC, max BTC, min BTC, BTC vs gold, BTC spreads, BTC totals, etc.
- No consistent pattern across diverse contract types
- Model learns: "sometimes do A, sometimes do B, sometimes do nothing"
- Result: Confused policy, negative performance

**Specific category success:**
- KXBTCD contains: ONLY daily Bitcoin price contracts
- Consistent pattern across all contracts
- Model learns: "daily BTC behaves like THIS, quote accordingly"
- Result: Clear policy, positive performance (+$13.45)

---

## Proposed Taxonomy Redesign

### Approach 1: Category-Group Agents (Recommended)

Train **specialized agents** for homogeneous category groups instead of one universal agent.

#### Group 1: Tennis Markets
- **Categories:** KXATP, KXATPCHALLENGERMATCH, KXATPGRANDSLAM, KXATPSETWINNER, etc.
- **Homogeneity:** All tennis match/set contracts, same sport dynamics
- **Agent:** Tennis MM Agent
- **Expected benefit:** Better than universal agent due to specialization
- **Training time:** ~10-20 categories, faster than 2,327

#### Group 2: Football/Soccer Markets
- **Categories:** KXAFCCLGAME, KXAPFDDH, KXAPFDDHGAME, KXALEAGUEGAME, etc.
- **Homogeneity:** All football/soccer match contracts, same sport dynamics
- **Agent:** Football MM Agent
- **Expected benefit:** Specialization for football-specific patterns

#### Group 3: Bitcoin Specific Markets (NOT Generic)
- **Categories:** KXBTCD (daily), KXBTCMAX (max), KXBTCMIN (min), KXBTCY (yearly)
- **Homogeneity:** Each subcategory separate, NOT mixing all BTC together
- **Agents:**
  - BTC Daily Agent (KXBTCD only)
  - BTC Max Agent (KXBTCMAX, KXBTCMAXMON, KXBTCMAXY)
  - BTC Min Agent (KXBTCMIN, KXBTCMINMON, KXBTCMINY)
- **Explicitly exclude:** KX (generic catch-all), KXB (generic catch-all)
- **Expected benefit:** Each agent specializes in specific BTC contract type

#### Group 4: Basketball Markets
- **Categories:** KXBBLGAME, KXBBSERIEAGAME, etc.
- **Homogeneity:** All basketball game contracts
- **Agent:** Basketball MM Agent

#### Group 5: Political Markets
- **Categories:** GOVPARTY*, KXAPRPOTUS, KXBERNIEMENTION, etc.
- **Homogeneity:** All political/election contracts
- **Agent:** Political MM Agent
- **Note:** Higher variance than sports, may need different hyperparameters

#### Group 6: Entertainment/Music Markets
- **Categories:** Album releases, chart positions, awards
- **Homogeneity:** All entertainment industry contracts
- **Agent:** Entertainment MM Agent
- **Note:** Currently poor performance, may not deploy even with specialization

### Taxonomy Hierarchy

```
Market-Making Agents
│
├─ Sports Agents
│  ├─ Tennis Agent (ATP, Challenger, Grand Slams)
│  ├─ Football Agent (League games, division games)
│  ├─ Basketball Agent (League games, series)
│  ├─ Baseball Agent
│  └─ General Sports Agent (catchall for minor sports)
│
├─ Crypto Agents
│  ├─ BTC Daily Agent (KXBTCD only)
│  ├─ BTC Max Agent (max price contracts)
│  ├─ BTC Min Agent (min price contracts)
│  ├─ BTC Yearly Agent (yearly contracts)
│  └─ BTC vs Other Agent (BTC vs gold, etc.)
│
├─ Political Agents
│  ├─ Election Agent (governor, senate, president)
│  ├─ Legislation Agent (bills, votes)
│  └─ Political Events Agent (mentions, announcements)
│
├─ Financial Agents
│  ├─ Fed Agent (Fed rate, policy)
│  ├─ Stock Agent (Apple, tech stocks)
│  └─ Commodity Agent (oil, gold)
│
└─ Entertainment Agents
   ├─ Music Agent (albums, charts)
   ├─ Awards Agent (Oscars, Grammys)
   └─ TV/Streaming Agent
```

---

## Training Strategy

### Phase 1: Train Group Agents (Recommended)

**For each category group:**

1. **Filter categories** to group members only
2. **Train specialized agent** on that group (500k steps)
3. **Evaluate** on held-out test set (last month of data)
4. **Deploy** if test performance positive

**Example: Tennis Agent**

```bash
python -m rl_bot.mm_train \
    --data output/rl_kalshi_trades_3mo.parquet \
    --markets output/rl_all_markets_3mo.parquet \
    --category-group "KX.*ATP.*" \  # Regex for tennis categories
    --timesteps 500000 \
    --ppo-epochs 10 \
    --run-name tennis_agent
```

**Benefits:**
- No catastrophic forgetting (each agent independent)
- Specialization improves performance
- Faster training (10-20 categories vs. 2,327)
- Can tune hyperparameters per group

**Costs:**
- More agents to maintain (10-15 vs. 1)
- Slightly higher storage (10-15 checkpoints vs. 1)
- Need to route categories to correct agent in production

---

### Phase 2: Category-Specific Agents (Maximum Performance)

**For highest-value categories, train individual agents:**

- KXBTCD Agent (daily BTC only)
- KXACBGAME Agent (specific game type)
- KXATP Agent (ATP main tour)

**Benefits:**
- Maximum specialization
- Highest performance
- Can tune hyperparameters per category

**Costs:**
- Many agents (108+ for all deployed categories)
- Higher maintenance burden
- Longer total training time

**Recommendation:** Use for top 10 revenue-generating categories only

---

## Deployment Architecture

### Option A: Multi-Agent Router

**Live trading system:**

```python
# Category → Agent mapping
AGENT_MAP = {
    "KXATP.*": tennis_agent,
    "KXATPCHALLENGERMATCH.*": tennis_agent,
    "KXBTCD.*": btc_daily_agent,
    "KXACBGAME.*": football_agent,
    # ...
}

def get_agent(ticker):
    for pattern, agent in AGENT_MAP.items():
        if re.match(pattern, ticker):
            return agent
    return None  # No agent for this category
```

**Benefits:**
- Clean separation of concerns
- Easy to add/remove agents
- Can update one agent without affecting others

---

### Option B: Ensemble (Advanced)

**For categories in multiple groups:**

```python
# Use multiple agents and combine predictions
btc_pred = btc_agent.predict(obs)
general_pred = general_agent.predict(obs)

# Weighted combination
final_pred = 0.7 * btc_pred + 0.3 * general_pred
```

**Benefits:**
- Robust to individual agent failures
- Can leverage multiple perspectives

**Costs:**
- More complex
- Higher latency (multiple inference calls)

---

## Category Filtering Rules

### Exclude from Training

**Generic catch-alls (too heterogeneous):**
- ❌ KX (all BTC contracts)
- ❌ KXB (BTC subcategory catch-all)
- ❌ Any category with >5,000 tickers (likely too generic)

**Illiquid categories:**
- ❌ Categories with <5 liquid tickers (>50 trades each)
- ❌ Categories with <500 total trades

**Poor performance categories:**
- ❌ Music/entertainment categories (KXBBCHARTPOSITIONALBUM, etc.)
- ❌ Categories with negative training performance despite specialization

### Include in Training

**Specific, homogeneous categories:**
- ✅ KXBTCD (daily BTC)
- ✅ KXATP (ATP tennis)
- ✅ KXACBGAME (specific game type)
- ✅ All sports categories with >10 liquid tickers

**Rule of thumb:** If category name includes specific time period, level, or type → include

---

## Data Split Strategy

### Temporal Train/Val/Test Split

**Avoid overfitting:**

| Split | Time Period | Purpose | Size |
|-------|-------------|---------|------|
| Train | Months 1-2 | Model training | 66% |
| Validation | First 2 weeks of month 3 | Hyperparameter tuning | 17% |
| Test | Last 2 weeks of month 3 | Final evaluation | 17% |

**Critical:**
- Never train on test set
- Only deploy if test performance positive
- Test set measures generalization

---

## Training Hyperparameters by Group

### Sports Categories (Lower variance)

```python
config = MMConfig(
    learning_rate=3e-4,
    gamma=0.99,
    ppo_epochs=10,
    batch_size=64,
    max_inventory=20,
    quote_size=1,
)
```

### Crypto Categories (Higher variance)

```python
config = MMConfig(
    learning_rate=1e-4,  # Lower LR for stability
    gamma=0.95,  # Lower gamma for shorter horizon
    ppo_epochs=5,   # Fewer epochs to prevent overfitting
    batch_size=128,  # Larger batch for variance reduction
    max_inventory=10,  # Lower inventory limit (more volatile)
    quote_size=1,
)
```

### Political Categories (Highest variance)

```python
config = MMConfig(
    learning_rate=1e-4,
    gamma=0.90,  # Much shorter horizon (news-driven)
    ppo_epochs=5,
    batch_size=128,
    max_inventory=5,  # Very conservative (unpredictable)
    quote_size=1,
)
```

---

## Implementation Roadmap

### Week 1-2: Group Definition
- ✅ Analyze 2,327 categories
- ✅ Define 10-15 category groups (see hierarchy above)
- ✅ Create category → group mapping
- ✅ Filter out generic catch-alls (KX, KXB)

### Week 3-4: Tennis Agent (Pilot)
- Train tennis agent on all ATP/Challenger categories
- Temporal train/val/test split
- Evaluate on test set
- If positive → Deploy in Phase 3
- If negative → Investigate, adjust hyperparameters

### Week 5-6: BTC Agents
- Train BTC Daily agent (KXBTCD only)
- Train BTC Max agent (KXBTCMAX, KXBTCMAXMON, KXBTCMAXY)
- Train BTC Min agent (KXBTCMIN, KXBTCMINMON, KXBTCMINY)
- Evaluate all on test sets
- Deploy KXBTCD if positive (already proven in current training)

### Week 7-8: Football Agent
- Train football agent on all soccer/football categories
- Evaluate
- Deploy if positive

### Week 9-12: Remaining Groups
- Basketball, political, financial, entertainment agents
- Train, evaluate, deploy profitable groups

### Month 4+: Category-Specific Agents
- For top 10 revenue categories
- Train individual agents
- Compare vs. group agents
- Deploy if >20% improvement over group agent

---

## Success Metrics

### Group Agent vs. Universal Agent

**Comparison metrics:**

| Metric | Universal Agent | Group Agent | Target |
|--------|----------------:|------------:|-------:|
| Mean reward | $6.16 | TBD | >$8.00 |
| Profitable % | 64% | TBD | >70% |
| Generic catch-all performance | Negative | Excluded | N/A |
| Training time | ~80 hours | ~10 hours | <50% |

**Success criteria:**
- Group agent mean reward >30% higher than universal agent
- No catastrophic forgetting (all groups maintain performance)
- Faster training time

---

## Risk Mitigation

### Risk: Group Agent Still Negative

**If tennis agent performs poorly despite specialization:**

**Possible causes:**
- Group still too heterogeneous (ATP main vs. Challenger are different)
- Sport-specific issues (tennis markets unpredictable)
- Hyperparameters need tuning

**Mitigation:**
1. Split into sub-groups (ATP Main Agent, Challenger Agent)
2. Tune hyperparameters (lower learning rate, more epochs)
3. Add domain-specific features (player rankings, surface type)
4. If still negative, exclude from deployment

### Risk: Too Many Agents to Maintain

**If 15 agents becomes burdensome:**

**Mitigation:**
1. Focus on profitable groups only (sports, BTC daily)
2. Use shared infrastructure (same code, different checkpoints)
3. Automate retraining pipeline
4. Only maintain agents for deployed categories

---

## Comparison: Universal vs. Group vs. Category-Specific

| Approach | Agents | Training Time | Performance | Maintenance | Use Case |
|----------|-------:|--------------|-------------|-------------|----------|
| **Universal** | 1 | 80 hours | Poor on catch-alls | Easy | ❌ Not recommended |
| **Group** | 10-15 | 10-20 hours | Good specialization | Medium | ✅ Recommended for deployment |
| **Category-specific** | 100+ | 40-60 hours | Best performance | Hard | ✅ For top 10 categories only |

---

## Recommended Next Steps

### Immediate (Week 1)

1. ✅ Document generic catch-all problem (done)
2. ✅ Update deployment plan to exclude KX, include KXBTCD (done)
3. Create category → group mapping file
4. Filter categories: exclude KX, KXB, illiquid categories

### Short-term (Weeks 2-4)

1. Train **Tennis Agent** (pilot program)
   - All KXATP* and KXATPCHALLENGERMATCH* categories
   - Temporal train/val/test split
   - Evaluate on test set
   - Compare vs. universal agent results

2. Train **BTC Daily Agent** (high-value)
   - KXBTCD only (already proven positive)
   - Should outperform universal agent's $13.45

### Medium-term (Months 2-3)

1. Train remaining group agents (football, basketball, BTC sub-types)
2. Evaluate all agents on test sets
3. Deploy profitable groups in production
4. Monitor live performance vs. backtest

### Long-term (Months 4-6)

1. Train category-specific agents for top 10 revenue categories
2. Implement ensemble approach (multiple agents per category)
3. Add domain-specific features (player rankings, market sentiment)
4. Continuous retraining on updated data (monthly)

---

## Conclusion

**Current universal agent approach fails on generic catch-alls** (KX: -$1.91, KXB: -$1.75)

**Group-specific agents solve this by:**
- Excluding generic catch-alls entirely
- Training specialized agents on homogeneous category groups
- Preventing catastrophic forgetting
- Enabling hyperparameter tuning per group

**Expected outcome:**
- 30-50% improvement in mean reward over universal agent
- Better performance on all category types
- No more generic catch-all failures

**Recommended deployment:**
- ✅ Group agents for production (10-15 agents)
- ✅ Category-specific agents for top 10 categories
- ❌ Universal agent (deprecated)

---

**See also:**
- `docs/deployment_plan.md` — Deployment strategy using group agents
- `docs/kx_investigation.md` — Generic catch-all problem analysis
- `docs/category_analysis.md` — Category performance breakdown
