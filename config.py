import os
from dotenv import load_dotenv

load_dotenv()

# Auth
API_KEY = os.getenv("PROD_API_KEY")
KEY_PATH = os.getenv("PROD_KEY_PATH", "rsa_keys/kalshi_bot_v3.txt")

# WebSocket
#WS_URL = "wss://trading-api.kalshi.com/trade-api/ws/v2"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# Trading mode
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# Market
MARKET_PREFIX = os.getenv("MARKET_PREFIX", "KXBTC")

# Trade filter thresholds
MIN_EDGE = 0.05
BUY_PROB_THRESHOLD = 0.70
SELL_PROB_THRESHOLD = 0.30
VOL_RATIO_THRESHOLD = 2.0
RANGE_THRESHOLD = 0.05
MIN_VOLUME = 5
EDGE_VOL_THRESHOLD = 0.10

# Fill model params
FILL_D0 = 0.03
FILL_WEIGHTS = (0.4, 0.2, 0.2, 0.2)

# Quoting
GRID_OFFSETS = [0.01, 0.02, 0.03, 0.04, 0.05]
SUBPENNY_IMPROVEMENT = 0.001

# Fees
MAKER_FEE_RATE = 0.0175
TAKER_FEE_RATE = 0.07

# Volatility tracker
VOL_WINDOW = 100

# Reconnection
RECONNECT_BASE = 1
RECONNECT_MAX = 30
