"""
Configuration for Kalshi BTC market making bot.

All trading parameters, thresholds, and API settings centralized here.
Sensitive values (API key, private key path) loaded from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# AUTHENTICATION
# ============================================================================

# Kalshi API key (from .env: PROD_API_KEY)
API_KEY = os.getenv("PROD_API_KEY")

# Path to RSA private key file (from .env: PROD_KEY_PATH)
KEY_PATH = os.getenv("PROD_KEY_PATH", "rsa_keys/kalshi_bot_v3.txt")

# ============================================================================
# WEBSOCKET CONNECTION
# ============================================================================

# Kalshi WebSocket URL
# Production: wss://trading-api.kalshi.com/trade-api/ws/v2
# Elections API: wss://api.elections.kalshi.com/trade-api/ws/v2
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# ============================================================================
# TRADING MODE
# ============================================================================

# If True: only log trades, don't submit to Kalshi
# If False: actually submit orders (LIVE TRADING)
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# ============================================================================
# MARKET SELECTION
# ============================================================================

# Ticker prefix to filter for (e.g., "KXBTC" for Bitcoin markets)
MARKET_PREFIX = os.getenv("MARKET_PREFIX", "KXBTC")

# ============================================================================
# TRADE FILTER THRESHOLDS
# ============================================================================

# Minimum edge (model_prob - market_price) to consider trading
# Filters out trades with insufficient profit potential
MIN_EDGE = 0.05

# Buy only if model predicts >= this probability
BUY_PROB_THRESHOLD = 0.70

# Sell only if model predicts <= this probability
SELL_PROB_THRESHOLD = 0.30

# Minimum ratio of recent_vol / avg_vol to trade
# Filters for volatility spikes (more trading opportunity)
VOL_RATIO_THRESHOLD = 2.0

# Minimum price range (max - min) in window to trade
# Filters out stagnant markets
RANGE_THRESHOLD = 0.05

# Minimum recent trade count to trade
# Filters for market activity
MIN_VOLUME = 5

# Minimum edge × vol_ratio to trade
# Combined filter ensuring sufficient opportunity
EDGE_VOL_THRESHOLD = 0.10

# ============================================================================
# FILL PROBABILITY MODEL
# ============================================================================

# Characteristic distance for exponential decay in distance_score
# Smaller = more aggressive decay (only quote very close to market)
FILL_D0 = 0.03

# Weights for fill probability components:
# (distance, momentum, volume, proximity)
# Must sum to 1.0 for probability interpretation
FILL_WEIGHTS = (0.4, 0.2, 0.2, 0.2)

# ============================================================================
# QUOTE GENERATION
# ============================================================================

# Grid of price offsets from market price to evaluate
# E.g., [0.01, 0.02] means try prices at market ± 1¢, market ± 2¢
GRID_OFFSETS = [0.01, 0.02, 0.03, 0.04, 0.05]

# Sub-penny improvement to jump ahead in price-time priority queue
# Added to buy prices, subtracted from sell prices
SUBPENNY_IMPROVEMENT = 0.001

# ============================================================================
# FEES (from Kalshi fee schedule)
# ============================================================================

# Maker fee rate (provide liquidity with limit orders)
MAKER_FEE_RATE = 0.0175  # 1.75% of contracts × price × (1 - price)

# Taker fee rate (remove liquidity with market orders)
TAKER_FEE_RATE = 0.07    # 7% of contracts × price × (1 - price)

# ============================================================================
# VOLATILITY TRACKING
# ============================================================================

# Number of price updates to track in rolling window
# Larger = smoother metrics but slower to react
VOL_WINDOW = 100

# ============================================================================
# CONNECTION RESILIENCE
# ============================================================================

# Initial reconnection delay in seconds
RECONNECT_BASE = 1

# Maximum reconnection delay in seconds (exponential backoff cap)
RECONNECT_MAX = 30
