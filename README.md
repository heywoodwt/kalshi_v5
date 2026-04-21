# Kalshi BTC Market Making Bot

Real-time automated market making system for Bitcoin prediction markets on Kalshi. Connects via WebSocket, analyzes market conditions, and generates optimal limit orders based on model predictions and multi-stage filtering.

## Architecture

```
WebSocket Stream → Market Filter → Volatility Tracker → Model → Filter Pipeline → Quoting Engine → Orders
```

### Core Components

1. **WebSocket Client** (`authentication_to_kalshi/websocket_client.py`)
   - Maintains authenticated connection to Kalshi's WebSocket API
   - Receives live ticker, trade, and orderbook data
   - Auto-reconnects with exponential backoff

2. **Market Filter** (`market_selector/market_filter.py`)
   - Filters incoming data to only BTC markets (configurable prefix)
   - Tracks discovered markets to avoid duplicate processing

3. **Volatility Tracker** (`volatility.py`)
   - Maintains rolling window of price history per market
   - Calculates real-time metrics: volatility, range, momentum, volatility ratios
   - Single-pass O(n) calculation for efficiency

4. **Model Interface** (`model/model_interface.py`)
   - Predicts fair value probability for each market
   - Currently placeholder (returns market price)
   - Production: replace with trained ML model

5. **Filter Pipeline** (`main.py:_maybe_quote()`)
   - 8-stage filter eliminating non-trading opportunities
   - Each filter must pass for quote generation

6. **Quoting Engine** (`market_selector/quoting_engine.py`)
   - Generates optimal buy/sell quotes on price grid
   - Calculates expected value (EV) per candidate quote
   - Selects highest-EV quotes with positive expected profit

7. **Fill Probability Model** (`market_selector/fill_probability.py`)
   - Estimates order fill probability using 4 factors:
     - Distance from market price (exponential decay)
     - Momentum alignment (sigmoid)
     - Recent trading volume
     - Historical price proximity
   - Weighted combination produces probability in [0, 1]

8. **Fee Calculator** (`market_selector/fee_calculator.py`)
   - Computes Kalshi maker/taker fees
   - Maker: 1.75% of contracts × price × (1 - price)
   - Taker: 7% of contracts × price × (1 - price)

## Filter Pipeline

All 8 filters must pass for a market to generate quotes:

1. **Valid Price Data** - Market has valid price information
2. **Volatility Spike** - `vol_ratio >= 2.0` (recent vol / avg vol)
3. **Sufficient Range** - `price_range >= 0.05` (not stagnant)
4. **Adequate Volume** - `recent_trades >= 5` (liquid market)
5. **Model Ready** - Prediction model initialized
6. **Sufficient Edge** - `|model_prob - market_price| >= 0.05`
7. **Strong Signal** - `model_prob >= 0.70` (buy) or `<= 0.30` (sell)
8. **Combined Threshold** - `edge × vol_ratio >= 0.10`

Thresholds configurable in `config.py`.

## Installation

### Prerequisites

- Python 3.9+
- Kalshi API account with API key and RSA private key

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install websockets python-dotenv cryptography

# Create .env file with credentials
cat > .env << EOF
PROD_API_KEY=your_kalshi_api_key
PROD_KEY_PATH=rsa_keys/your_private_key.pem
PAPER_TRADING=true
MARKET_PREFIX=KXBTC
EOF

# Add your RSA private key to rsa_keys/
```

## Configuration

All parameters in `config.py`:

### Trading Mode
- `PAPER_TRADING=true` - Log trades only (safe mode)
- `PAPER_TRADING=false` - Submit actual orders (requires implementation)

### Market Selection
- `MARKET_PREFIX="KXBTC"` - Only trade BTC markets

### Filter Thresholds
- `MIN_EDGE=0.05` - Minimum model edge to trade
- `BUY_PROB_THRESHOLD=0.70` - Buy only if model >= 70%
- `SELL_PROB_THRESHOLD=0.30` - Sell only if model <= 30%
- `VOL_RATIO_THRESHOLD=2.0` - Min recent/avg volatility ratio
- `RANGE_THRESHOLD=0.05` - Min price range in window
- `MIN_VOLUME=5` - Min recent trade count
- `EDGE_VOL_THRESHOLD=0.10` - Min edge × vol_ratio

### Quote Generation
- `GRID_OFFSETS=[0.01, 0.02, 0.03, 0.04, 0.05]` - Price grid to evaluate
- `SUBPENNY_IMPROVEMENT=0.001` - Queue priority improvement

### Volatility Tracking
- `VOL_WINDOW=100` - Rolling window size for price history

### Connection
- `WS_URL` - Kalshi WebSocket endpoint
- `RECONNECT_BASE=1` - Initial reconnect delay (seconds)
- `RECONNECT_MAX=30` - Max reconnect delay (exponential backoff cap)

## Running

```bash
# Paper trading mode (logs only)
python main.py

# Live trading mode (requires order execution implementation)
PAPER_TRADING=false python main.py
```

## Output Example

```
2026-04-21 13:45:23 [INFO] kalshi: Starting Kalshi BTC trading system (PAPER mode)
2026-04-21 13:45:24 [INFO] websocket_client: Connected to wss://api.elections.kalshi.com/trade-api/ws/v2
2026-04-21 13:45:24 [INFO] websocket_client: Subscribed to global ticker channel
2026-04-21 13:45:31 [INFO] kalshi: Discovered BTC market: KXBTC-26APR21-100000

[QUOTE] KXBTC-26APR21-100000
  BUY: 0.441 (EV=0.0237, fill=0.687, edge=0.350)
```

## Project Structure

```
kalshi_v5/
├── main.py                          # Entry point, WebSocket handlers, quote logic
├── config.py                        # All configuration parameters
├── volatility.py                    # Rolling volatility tracker
├── authentication_to_kalshi/
│   ├── auth.py                      # RSA-PSS signing for Kalshi API
│   └── websocket_client.py          # WebSocket connection management
├── market_selector/
│   ├── market_filter.py             # Market prefix filtering
│   ├── quoting_engine.py            # Quote generation and EV calculation
│   ├── fill_probability.py         # Fill probability estimation
│   └── fee_calculator.py            # Kalshi fee calculation
├── model/
│   └── model_interface.py           # Prediction model (placeholder)
├── test_of_concept/                 # Testing and proof-of-concept scripts
├── notes/
│   └── test_results_v1.md          # Test results documentation
├── .env                             # API credentials (not committed)
└── rsa_keys/                        # RSA private keys (not committed)
```

## Development

### Testing

Test scripts in `test_of_concept/`:
- `test_trading_simulation.py` - Full pipeline simulation
- `test_order_generation.py` - Quote generation validation

### Enabling Live Trading

Currently in paper trading mode. To enable actual order submission:

1. Set `PAPER_TRADING=false` in `.env`
2. Implement REST API order submission in `main.py` around line 308-312
3. Add order placement function using Kalshi REST API:
   ```python
   POST /trade-api/v2/portfolio/orders
   {
     "ticker": "KXBTC-...",
     "side": "yes",
     "type": "limit",
     "yes_price": 44,  # cents
     "count": 1
   }
   ```

### Model Integration

Replace `model/model_interface.py` with actual prediction model:

1. Implement `predict()` method to return fair value probability
2. Use features: `market_price`, volatility metrics, external data
3. Set `_ready=True` when model initialization complete

Expected features dict:
```python
{
    "market_price": 0.45,
    "vol": 0.03,
    "range": 0.12,
    "vol_ratio": 2.33,
    "momentum": 0.02,
    "vol_up": 0.025,
    "vol_down": 0.028,
    "last_price": 0.45
}
```

## Performance Considerations

- **O(n) volatility calculation** - Single pass through price window
- **Lazy initialization** - Markets only initialized when first seen
- **Efficient filtering** - Early filters eliminate markets before expensive calculations
- **Single WebSocket connection** - Multiplexed across all markets
- **Deque-based windows** - Fixed memory per market

## Future Rust Migration

Code designed for easy translation to Rust:
- No pandas/numpy (except potential future ML model)
- Simple data structures (dicts, lists, deques)
- Minimal dynamic typing
- Explicit type hints throughout
- Functional decomposition with pure functions

## Risk Management

- **Inventory limits** - Won't add to existing positions (long blocks buys, short blocks sells)
- **Paper trading default** - Safe mode by default
- **Position tracking** - Per-market inventory maintained
- **Filter pipeline** - Multiple safety checks before trading
- **Fee awareness** - All quotes account for maker fees in EV calculation

## Notes

- System currently uses placeholder model (returns market price = no edge)
- Real trading requires:
  1. Trained prediction model
  2. REST API order submission implementation
  3. Proper risk management and position monitoring
- See `notes/test_results_v1.md` for detailed test results
- WebSocket auto-reconnects with exponential backoff for reliability
- All prices normalized to [0, 1] range internally
