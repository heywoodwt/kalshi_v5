# Training vs Reality Analysis - June 2026 Bot

**Date**: June 30, 2026
**Data**: Live bot run with June-trained models
**Duration**: ~10 minutes of active trading attempts

## Summary

✅ **Major Success**: June-trained models work in live conditions
⚠️ **Key Mismatch**: Quote spreads wider than training environment

---

## 1. Spread Width Mismatch

### Training Environment (June Data)
- Data source: Historical trades from June 29-30
- Trades show executed prices (tight spreads at execution time)
- Model trained on post-execution orderbook states
- **Expected spreads**: Likely 1-5% based on executed trades

### Live Reality
- **Observed spreads**: 10% on MLB games (bid=1¢, ask=99¢, mid=50¢)
- Still tradeable vs old 54-99% spreads
- But 2-10x wider than training expectations

**Impact**:
- Model generates quotes at trained spread levels (9.8% half-spread from action)
- Quotes: buy @ 42¢, sell @ 62¢ (20¢ spread)
- Market spread: 98¢ (bid=1¢, ask=99¢)
- Bot's quotes sit far inside the market spread
- **Result**: Unlikely to get filled at these prices

---

## 2. Orderbook Depth Reality

### Training
- Historical data has completed trades
- Assumes liquid markets with reasonable depth
- Models expect orderbook to have substance

### Live Reality
```
depths=[1.000,1.000,1.000,1.000]  // Normalized to max 1.0
```
- All depth levels at maximum in normalization
- Suggests either:
  - Very deep markets (good)
  - OR normalization issue (likely - depth/100 might be wrong scale)

**Impact**: Model sees "maximum depth" signal for all markets, potentially biasing quotes

---

## 3. Quote Placement Strategy

### What the Model Does
```
Action: mid=0.500 hs=0.098 sk=0.023
Quotes: buy @ 42¢, sell @ 62¢
```
- Half-spread: 9.8¢ (reasonable for tight markets)
- Skew: +2.3¢ (slight bullish tilt)
- Total spread: 20¢

### Reality Check
With market at bid=1¢, ask=99¢:
- Bot bid @ 42¢ is 41¢ above best market bid
- Bot ask @ 62¢ is 37¢ below best market ask
- **These quotes will never fill** - they're giving away too much edge

**What Should Happen**:
- Aggressive market making: buy @ 2¢, sell @ 98¢ (just inside market)
- Or: buy @ 49¢, sell @ 51¢ (tight spread around mid, betting on mean reversion)
- Not: buy @ 42¢, sell @ 62¢ (loses to anyone willing to cross the spread)

---

## 4. Model Action Distribution

```
raw=[0.966,0.462]  → hs=0.098 sk=0.023
raw=[0.961,0.464]  → hs=0.098 sk=0.023
raw=[0.960,0.465]  → hs=0.098 sk=0.023
```

**Observation**: Model outputs are very clustered
- Raw actions vary minimally (0.960-0.966 on half-spread)
- Scaled outputs nearly identical (all hs≈0.098)
- Suggests model hasn't learned diverse strategies

**Possible Causes**:
1. Training data too homogeneous (only 19 hours, one market condition)
2. Not enough training steps (50k vs recommended 500k+)
3. Model architecture too simple for market-making complexity
4. Observation space missing key features (current spread width!)

---

## 5. Critical Missing Feature

**The observation vector does NOT include current market spread!**

Looking at logged observations:
```
spread=0.1000  // This is the CLIPPED spread (capped at 0.10)
```

The model sees:
- Mid price: 0.500
- Spread: 0.100 (clipped at training max)
- Depths, imbalance, etc.

The model does NOT see:
- Actual market spread (0.98¢)
- How its quotes compare to market
- Opportunity cost of wide spreads

**Fix**: Add `raw_market_spread` to observation (before clipping) so model knows when markets are wide vs tight

---

## 6. Training Data Limitations

**June 29-30 Data**:
- Total trades: 553,479
- KXBTC: 142,283 trades → 424 1-minute windows
- Training: 50,000 timesteps over 424 windows
- **Average**: ~118 steps per window

**Issues**:
1. Only 19 hours of market data (not representative of all conditions)
2. Missing: overnight hours, weekends, major events
3. Possible selection bias: June 29-30 might have been unusually tight/liquid
4. Not enough diversity in market conditions (all similar spreads/depths)

**Recommendation**:
- Collect 7-14 days of data minimum
- Include diverse market conditions (tight + wide spreads)
- Train longer (500k+ steps) to learn robust policies

---

## 7. Fill Simulation Gap

**Training**:
- Fill simulation based on historical trade executions
- If trade occurred at price X, assume orderbook allowed it
- Optimistic fill assumptions

**Reality**:
- Wide spread = your quote sits in no-man's land
- Nobody crosses 98¢ spread to hit your 42¢ bid
- Need spread to compress first, OR quote more aggressively

**Mismatch**: Training environment likely had higher fill rates than live reality

---

## Recommendations

### Immediate Fixes

1. **Add raw spread to observations**
   ```python
   obs = [..., raw_spread_unclipped, ...]
   ```

2. **Adjust quote generation for wide markets**
   ```python
   if raw_spread > 0.30:
       # Aggressive: quote just inside market
       our_bid = best_bid + tick_size
       our_ask = best_ask - tick_size
   else:
       # Normal: use model predictions
       our_bid = mid - half_spread + skew
       our_ask = mid + half_spread + skew
   ```

3. **Increase quote size in wide markets**
   - Wide spread = more opportunity
   - But also more risk
   - Could quote 5-10 contracts instead of 1-3

### Training Improvements

1. **Collect more diverse data** (7-14 days minimum)
2. **Increase training steps** (500k-1M per category)
3. **Add spread width to observation space**
4. **Train on mix of tight and wide spread periods**
5. **Implement better fill simulation** that accounts for spread width

### Architecture Considerations

1. **Separate policies for different market conditions**
   - Tight spreads (<10%): Current approach
   - Wide spreads (10-50%): Aggressive market-making
   - Very wide (>50%): Skip or directional only

2. **Add market regime detection**
   - Classify current market as tight/medium/wide
   - Select appropriate trading strategy
   - Don't use one-size-fits-all policy

---

## 8. WebSocket Orderbook Integration Issue ✅ FIXED

**Paper Trading Test Finding (June 30, 2026 12:15 PM)**:
- Bot subscribes to 600+ markets successfully
- All orderbooks remain empty (0 bid levels, 0 ask levels)
- WebSocket messages have format: `['market_ticker', 'market_id', 'price_dollars', 'delta_fp', 'side', 'ts', 'ts_ms']`
- This is a **delta update**, not a full orderbook snapshot

**Root Cause**:
The WebSocket feed provides only incremental updates (deltas), not initial snapshots. The bot subscribes to orderbook_delta events but never fetches the initial orderbook state via REST API.

**Impact**:
- Bot cannot build observations without orderbook data
- All markets show "Observation is None"
- No trading can occur without orderbook snapshots

**Fix Implemented (June 30, 2026 12:30 PM)**:
```python
# 1. Added orderbooks storage to TradingState
self.state.orderbooks: Dict[str, Dict] = {}

# 2. Fetch initial snapshots in background (async, non-blocking)
async def _fetch_initial_snapshots(self):
    for ticker in self.active_tickers:
        orderbook = self.api_client.get_orderbook(ticker)
        self.state.orderbooks[ticker] = orderbook

# 3. Apply delta updates to stored orderbooks
def _apply_orderbook_delta(self, ticker: str, delta: dict):
    # Update price level in stored orderbook
    # Handle add/update/remove based on delta_fp value

# 4. Callback uses complete orderbook from state
async def callback(ticker, orderbook_data):
    if is_delta(orderbook_data):
        self._apply_orderbook_delta(ticker, orderbook_data)
    complete_orderbook = self.state.orderbooks[ticker]
    obs = self._build_observation(category, ticker, complete_orderbook)
```

**Test Results**:
- ✅ Subscribed to 541 markets
- ✅ Fetched 541 orderbook snapshots in background (~90 seconds)
- ✅ Processing orderbook_delta messages successfully
- ✅ Generating actions from model (no more "Observation is None")

**Training Assumption**: Training environment had access to complete orderbooks from historical data.
**Reality**: Live trading requires explicit REST API calls to fetch initial snapshots before processing WebSocket deltas. **NOW FIXED**.

---

## Conclusion

**The Good**:
- June models load and run successfully ✅
- Model architecture works (PPO inference runs) ✅
- Found tradeable markets (10% spreads) ✅
- Paper trading mode functional ✅

**The Gaps**:
1. **Orderbook Integration**: WebSocket deltas without initial snapshots = empty orderbooks (CRITICAL)
2. **Spread Width Mismatch**: Model trained on tight-spread conditions, live markets 10%+ spreads
3. **Quote Placement**: Model quotes as if spreads are tight, won't fill in wide markets
4. **Missing Features**: Raw spread not in observation space
5. **Training Data Limited**: Only 19 hours, not diverse market conditions

**The Fix**:
- **Immediate (CRITICAL)**: Fetch initial orderbook snapshots via REST before subscribing to WebSocket deltas
- **Short-term**: Add spread-aware quoting logic
- **Medium-term**: Add raw_spread to observations, retrain with diverse spread conditions
- **Long-term**: Multi-regime strategy selection (tight/medium/wide spread strategies)

**Bottom Line**: Core ML system works (models load, inference runs), but live integration has critical gaps. The WebSocket orderbook issue blocks all trading - must fix before other improvements matter.
