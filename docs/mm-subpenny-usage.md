# MM Subpenny Pricing Usage Guide

## Overview

The market-making bot now supports subpenny pricing (+/-0.001 adjustments) for queue priority on Kalshi markets that support fine tick sizes.

Subpenny pricing is a critical advantage in market making: by quoting at prices just ahead of competitors (bid +0.001, ask -0.001), you improve fill probability without sacrificing price quality. This is especially valuable for binary options markets where even 0.1% improvements compound over thousands of trades.

### Why Queue Priority Matters

In Kalshi's order book matching:
- Orders at the same price level are matched in FIFO order
- Quoting +0.001 on the bid puts you ahead of all 0.01-tick limit orders
- Quoting -0.001 on the ask ensures your sell order fills before competitors
- Result: Higher fill rates → Lower opportunity cost → Better PnL

## Features

- **Queue Priority**: Bid +0.001, Ask -0.001 to jump ahead of other market makers
- **Market-Aware**: Validates tick sizes (deci_cent, tapered_deci_cent, linear_cent)
- **Orderbook Integration**: 16-dim observation space with multi-level depth
- **Dual-Mode**: Works in training (parquet) and production (API/WebSocket)
- **Automatic Validation**: Prevents invalid subpenny quotes on linear_cent markets

## Training Mode (HPC)

### Data Requirements

1. **Trades**: `output/rl_kalshi_trades_3mo.parquet`
   - Columns: ticker, yes_price, count, taker_side, created_time
   - Used to compute VWAP for mid price fallback

2. **Markets**: `output/rl_all_markets_3mo.parquet`
   - Must include `price_level_structure` column
   - Valid values: "linear_cent", "deci_cent", "tapered_deci_cent"
   - Used to validate subpenny quotes per market

3. **Orderbooks** (optional): `output/mm_orderbooks.parquet`
   - Columns: ticker, created_time, bid, ask, bid_size, ask_size
   - Provides 3-level depth for better feature engineering
   - If missing, environment uses fallback values (see Observation Space)

### Example Usage

```python
import polars as pl
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader

# Load data
trades_df = pl.read_parquet("output/rl_kalshi_trades_3mo.parquet")
markets_df = pl.read_parquet("output/rl_all_markets_3mo.parquet")

# Initialize metadata loader for tick validation
metadata_loader = MarketMetadataLoader(
    mode="parquet",
    parquet_path="output/rl_all_markets_3mo.parquet",
)

# Preprocess data into time-aligned windows
ticker_data = preprocess_mm_data(trades_df)

# Create environment with subpenny enabled
cfg = MMConfig(
    subpenny_enabled=True,
    max_inventory=100,
    quote_size=10,
)
env = MMEnv(
    ticker_data=ticker_data,
    cfg=cfg,
    metadata_loader=metadata_loader,
)

# Train agent with PPO
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Preprocessing Pipeline

The `preprocess_mm_data()` function handles data alignment:

```python
def preprocess_mm_data(
    trades_df: pl.DataFrame,
    orderbooks_df: pl.DataFrame | None = None,
    window_size_s: int = 60,
) -> dict[str, list[dict]]:
    """
    Convert raw trade/orderbook data into environment-ready windows.

    Args:
        trades_df: Raw trades from Kalshi API
        orderbooks_df: Optional orderbook snapshots (aligned to trade windows)
        window_size_s: Time window size in seconds (default 60s)

    Returns:
        Dict mapping ticker -> list of windows
        Each window contains: {"trades": [...], "orderbook": {...}, "timestamp": ...}
    """
```

## Configuration

### MMConfig Parameters

All configuration lives in `rl_bot/mm_config.py`:

```python
@dataclass(frozen=True)
class MMConfig:
    # Position limits
    max_inventory: int = 100          # Max contracts to hold
    quote_size: int = 10              # Size per quote leg

    # API configuration (for live trading)
    api_environment: str = "demo"     # "demo" or "production"
    api_base_url: str = ""            # Auto-set if empty
    ws_url: str = ""                  # Auto-set if empty

    # Subpenny pricing
    subpenny_enabled: bool = True     # Enable/disable subpenny adjustment

    # Balance monitoring
    balance_check_interval_s: int = 60    # Check balance every N seconds
    min_balance_cents: float = 100.0      # Minimum balance to allow trading
```

### API Endpoints

**Demo** (for paper trading):
- REST API: `https://external-api.demo.kalshi.co/trade-api/v2`
- WebSocket: `wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2`

**Production** (for live trading):
- REST API: `https://external-api.kalshi.com/trade-api/v2`
- WebSocket: `wss://external-api-ws.kalshi.com/trade-api/ws/v2`

Endpoints are auto-configured in `MMConfig.__post_init__()`:

```python
cfg = MMConfig(api_environment="production")
# api_base_url automatically set to production REST endpoint
# ws_url automatically set to production WebSocket endpoint
```

## Market Metadata Loader

The `MarketMetadataLoader` class handles tick size validation across different Kalshi market types.

### Initialization (Parquet Mode)

```python
from rl_bot.mm_metadata import MarketMetadataLoader

metadata_loader = MarketMetadataLoader(
    mode="parquet",
    parquet_path="output/rl_all_markets_3mo.parquet",
)

# Load metadata for specific tickers
metadata = metadata_loader.load_metadata([
    "KXBTC.0625.25",
    "KXBTC.0725.25",
])
```

### Market Metadata Structure

Each market has a `MarketMetadata` object with:

```python
@dataclass
class MarketMetadata:
    ticker: str                        # e.g., "KXBTC.0625.25"
    price_level_structure: str         # "linear_cent", "deci_cent", "tapered_deci_cent"
    tick_size_low: float              # Tick size at prices < 0.10
    tick_size_mid: float              # Tick size at prices 0.10-0.90
    tick_size_high: float             # Tick size at prices > 0.90
```

### Tick Size Validation

Use `get_valid_tick_size()` to get the tick size at any price:

```python
# For linear_cent market (only 0.01 ticks)
tick = metadata_loader.get_valid_tick_size("KXBTC.0625.25", price=0.50)
# Returns: 0.01

# For deci_cent market (only 0.001 ticks everywhere)
tick = metadata_loader.get_valid_tick_size("KXBTC.0725.25", price=0.50)
# Returns: 0.001

# For tapered_deci_cent market (context-dependent)
tick = metadata_loader.get_valid_tick_size("KXBTC.0825.25", price=0.05)
# Returns: 0.001 (at tail)

tick = metadata_loader.get_valid_tick_size("KXBTC.0825.25", price=0.50)
# Returns: 0.01 (at middle)
```

### Subpenny Support Check

Use `supports_subpenny()` to check if a market allows 0.001 tick pricing:

```python
# Check if subpenny (0.001) is valid at this price
if metadata_loader.supports_subpenny("KXBTC.0625.25", price=0.50):
    # Can apply subpenny adjustment
    adjusted_bid = base_bid + 0.001
else:
    # Must use base price (market only supports 0.01)
    adjusted_bid = base_bid
```

## Observation Space (16 dimensions)

All observations are float32 values normalized to ranges for neural network stability.

### Feature Descriptions

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | mid_price | [0.01, 0.99] | Current mid price (midpoint of bid/ask spread or VWAP) |
| 1 | spread | [0.01, 0.10] | Bid-ask spread width |
| 2 | bid_depth_l0 | [0, 1] | Best bid level normalized depth (min(contracts/100, 1)) |
| 3 | ask_depth_l0 | [0, 1] | Best ask level normalized depth |
| 4 | bid_depth_l1 | [0, 1] | Second-level bid depth |
| 5 | ask_depth_l1 | [0, 1] | Second-level ask depth |
| 6 | bid_depth_l2 | [0, 1] | Third-level bid depth |
| 7 | ask_depth_l2 | [0, 1] | Third-level ask depth |
| 8 | book_imbalance | [-1, 1] | Orderbook imbalance (bid_volume - ask_volume) / (bid_volume + ask_volume) |
| 9 | trade_volume_1m | [0, 1] | Recent trade volume in current window (normalized by 50 trades) |
| 10 | inventory_norm | [-1, 1] | Normalized inventory position (inventory / max_inventory) |
| 11 | unrealized_pnl_norm | [-1, 1] | Unrealized PnL normalized by inventory |
| 12 | tte_log | [0, ~4] | Log time to expiry (log(1 + hours)) |
| 13 | momentum | [-0.1, 0.1] | Price momentum (current_mid - mid_5_steps_ago) |
| 14 | realized_pnl_norm | [-1, 1] | Realized PnL from fills (normalized by 50x max_inventory) |
| 15 | fills_ratio | [0, 1] | Fill rate (total_fills / quote_size) |

### Building Observations

The environment automatically builds observations in `_build_obs()`:

```python
obs = env._build_obs()  # Returns np.array of shape (16,) dtype=float32
# All values clipped to observation_space bounds for safety
```

## API Reference

### Core Classes

#### MMConfig
Configuration dataclass for market-making environment.

```python
cfg = MMConfig(
    max_inventory=100,           # Max position size
    quote_size=10,               # Size per quote
    subpenny_enabled=True,       # Enable subpenny adjustment
    api_environment="demo",      # "demo" or "production"
    balance_check_interval_s=60, # Balance check frequency
    min_balance_cents=100.0,     # Minimum balance threshold
)
```

#### MarketMetadata
Immutable market metadata container.

```python
@dataclass
class MarketMetadata:
    ticker: str
    price_level_structure: str
    tick_size_low: float
    tick_size_mid: float
    tick_size_high: float
```

#### MarketMetadataLoader
Loads and caches market metadata with tick validation.

**Key methods:**

```python
# Load metadata for tickers
metadata = loader.load_metadata(["TICKER1", "TICKER2"])

# Get valid tick size at specific price
tick = loader.get_valid_tick_size("TICKER", price=0.50)

# Check if subpenny (0.001) is valid
if loader.supports_subpenny("TICKER", price=0.50):
    ...
```

#### OrderbookSnapshot
Extended with 3-level depth for observation engineering.

**Key methods:**

```python
class OrderbookSnapshot:
    def mid_price(self) -> float | None:
        """Return mid price or None if spread invalid."""

    def spread(self) -> float | None:
        """Return bid-ask spread or None."""

    def imbalance(self) -> float:
        """Return normalized order book imbalance in [-1, 1]."""

    # Multi-level depth attributes
    yes_size_l1: int | None      # Second-level bid
    no_size_l1: int | None       # Second-level ask
    yes_size_l2: int | None      # Third-level bid
    no_size_l2: int | None       # Third-level ask
```

#### MMEnv
Market-making reinforcement learning environment.

**Initialization:**

```python
env = MMEnv(
    ticker_data=dict[str, list[dict]],  # Output of preprocess_mm_data()
    cfg=MMConfig(...),                  # Configuration
    metadata_loader=MarketMetadataLoader(...),  # For tick validation
)
```

**Key methods:**

```python
# Gymnasium interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Action: np.array([half_spread_control, skew_control]) in [-1, 1]
# Output: 16-dim observation, reward, terminated flag, info dict

# Subpenny application
adjusted_price = env._apply_subpenny_adjustment(base_price, side="bid")
```

**Info dict contents:**

```python
{
    "ticker": str,                  # Current market ticker
    "timestamp": datetime,          # Current window timestamp
    "bid": float,                   # Actual bid quote (3 decimals)
    "ask": float,                   # Actual ask quote (3 decimals)
    "inventory": int,               # Current position
    "fills_buy": int,               # Buy fills this episode
    "fills_sell": int,              # Sell fills this episode
    "pnl": float,                   # Total PnL (realized + unrealized)
    "realized_pnl": float,          # PnL from closed positions
    "unrealized_pnl": float,        # PnL from open position
    "half_spread": float,           # Applied half-spread (0.01-0.10)
    "skew": float,                  # Applied skew (-0.05 to 0.05)
    "mid_price": float,             # Current mid price
}
```

### Helper Functions

#### preprocess_mm_data
Converts raw trades/orderbooks into environment-ready windows.

```python
def preprocess_mm_data(
    trades_df: pl.DataFrame,
    orderbooks_df: pl.DataFrame | None = None,
    window_size_s: int = 60,
) -> dict[str, list[dict]]:
    """
    Process raw market data into time-aligned windows.

    Args:
        trades_df: Trades with columns: ticker, yes_price, count,
                   taker_side, created_time
        orderbooks_df: Optional orderbook snapshots
        window_size_s: Window duration in seconds (default 60)

    Returns:
        Dict[ticker] -> List of dicts with keys:
        - "trades": list of trade dicts (yes_price, count, taker_side)
        - "orderbook": OrderbookSnapshot or None
        - "timestamp": datetime of window
    """
```

#### scale_action
Converts RL action to bid/ask parameters.

```python
def scale_action(
    action: np.ndarray,  # [half_spread_control, skew_control] in [-1, 1]
    cfg: MMConfig,
) -> tuple[float, float]:
    """
    Returns:
        (half_spread, skew) where:
        - half_spread: [0.01, 0.10] (spread width / 2)
        - skew: [-0.05, 0.05] (asymmetric adjustment)
    """
```

## Testing

### Test Files

All tests in `tests/test_mm_*.py`:

- `test_mm_config.py` - Configuration validation (3 tests)
- `test_mm_metadata.py` - Metadata loading and tick validation (4 tests)
- `test_mm_orderbook_integration.py` - Orderbook snapshot multi-level depth (4 tests)
- `test_mm_subpenny.py` - Subpenny adjustment logic (4 tests)
- `test_mm_observation_space.py` - 16-dim observation generation (2 tests)
- `test_mm_step_integration.py` - Full step() function (2 tests)
- `test_mm_full_integration.py` - End-to-end training scenario (1 test)

**Total: 20 tests**

### Running Tests

```bash
# Run all market-making tests
pytest tests/test_mm_*.py -v

# Run specific test suite
pytest tests/test_mm_config.py -v
pytest tests/test_mm_subpenny.py -v

# Run with short traceback for failures
pytest tests/test_mm_*.py -v --tb=short

# Run with coverage
pytest tests/test_mm_*.py --cov=rl_bot --cov-report=html
```

### Example Test

```python
def test_subpenny_adjustment_deci_cent():
    """Test subpenny applied on deci_cent markets."""
    cfg = MMConfig(subpenny_enabled=True)
    loader = MarketMetadataLoader(mode="parquet", parquet_path="...")
    env = MMEnv(ticker_data=..., cfg=cfg, metadata_loader=loader)

    env._current_ticker = "TICKER_DECI"

    # Should apply +0.001 on bid
    adjusted_bid = env._apply_subpenny_adjustment(0.500, "bid")
    assert adjusted_bid == 0.501

    # Should apply -0.001 on ask
    adjusted_ask = env._apply_subpenny_adjustment(0.510, "ask")
    assert adjusted_ask == 0.509
```

## Troubleshooting

### Subpenny not applied

If your quotes are not getting subpenny adjustment, check:

1. **Config enabled**: Verify `subpenny_enabled=True` in `MMConfig`
   ```python
   cfg = MMConfig(subpenny_enabled=True)
   ```

2. **Metadata loader provided**: Ensure you pass `metadata_loader` to `MMEnv`
   ```python
   env = MMEnv(ticker_data=..., cfg=cfg, metadata_loader=loader)
   # Not: env = MMEnv(ticker_data=..., cfg=cfg, metadata_loader=None)
   ```

3. **Market supports subpenny**: Check market's `price_level_structure`
   ```python
   # Check in parquet file
   markets_df.filter(pl.col("ticker") == "YOUR_TICKER")
   # Look for: "deci_cent" or "tapered_deci_cent" in price_level_structure
   # "linear_cent" will NOT support subpenny
   ```

4. **Price in valid range**: For tapered markets, subpenny only works at tails
   ```python
   # For tapered_deci_cent:
   # - price < 0.10: supports 0.001
   # - price 0.10-0.90: only 0.01 (middle)
   # - price > 0.90: supports 0.001
   ```

### Observation shape mismatch

If you get "Observation has wrong shape (10,) vs (16,)":

**Cause**: Old checkpoints trained on 10-dim observation space

**Solution**:
- Retrain agent with current 16-dim space
- Delete old checkpoint files
- Use checkpoints created after this feature

```bash
# Check checkpoint observation size
python -c "import pickle; ckpt=pickle.load(open('your.pkl','rb')); print(ckpt['observation_space'])"
```

### Metadata loader errors

**"parquet_path required for parquet mode"**

Ensure you pass path to markets parquet file:
```python
loader = MarketMetadataLoader(
    mode="parquet",
    parquet_path="output/rl_all_markets_3mo.parquet",  # Must not be None
)
```

**"No metadata for TICKER"**

Market not found in parquet file. Check:
1. Ticker spelling in data
2. Markets parquet file contains that ticker
3. Use `pl.read_parquet(...).select("ticker").unique()` to list available tickers

## API Mode (Future)

Production trading requires API-based market metadata fetching:

```python
# FUTURE: API mode (not yet implemented)
loader = MarketMetadataLoader(
    mode="api",
    api_key="your_kalshi_api_key",
    api_secret="your_kalshi_secret",
)
```

See: Task 10 implementation plan for API metadata loader.

## WebSocket Integration (Future)

Live trading will use `MMEnvLive` with WebSocket orderbook updates:

```python
# FUTURE: Live environment with WebSocket streaming
env = MMEnvLive(
    cfg=MMConfig(api_environment="production"),
    metadata_loader=loader,
    ws_url="wss://...",
)

# Automatically updates orderbooks from WebSocket stream
obs, info = env.reset()
obs, reward, terminated, _, info = env.step(action)
```

See: Task 11 implementation plan for MMEnvLive.

## Next Steps

1. **Retrain PPO agent** with 16-dim observation space
2. **Backtest** on validation set (compare subpenny vs baseline)
3. **Implement API mode** for production metadata (Task 10)
4. **Add MMEnvLive** class with WebSocket (Task 11)
5. **Paper trade** on demo environment
6. **Deploy** to production with balance monitoring

## References

- **Architecture Spec**: See `.superpowers/sdd/mm-subpenny-spec.md`
- **Kalshi API Docs**: https://kalshi.com/api
- **Binary Options**: https://www.investopedia.com/terms/b/binary-option.asp
- **Market Microstructure**: https://en.wikipedia.org/wiki/Market_microstructure
