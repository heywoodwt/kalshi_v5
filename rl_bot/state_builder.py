import math

import numpy as np

from rl_bot.btc_data import BTCDataPoller
from rl_bot.config import (
    ACTION_CLOSE_NO,
    ACTION_CLOSE_YES,
    ACTION_HOLD,
    RLConfig,
)
from rl_bot.reward import PnLTracker
from model.hp_dfm_rte.orderbook import OrderbookSnapshot

# Normalization constants for trade count and orderbook depth
_TRADE_COUNT_NORM = 50.0   # normalize trade count to ~[0, 1]
_DEPTH_NORM = 100.0        # normalize orderbook depth to ~[0, 1]


def build_state(
    ticker: str,
    vol_metrics: dict[str, float],
    orderbook: OrderbookSnapshot,
    btc_poller: BTCDataPoller,
    pnl_tracker: PnLTracker,
    market_price: float,
    time_to_expiry_h: float,
    trade_count: int,
    cfg: RLConfig,
) -> np.ndarray:
    """Build the 18-dimensional state vector for a single market.

    Feature order:
      0: market_price            [0, 1]
      1: bid_ask_spread          [0, ~0.2]
      2: recent_volatility       [0, ~0.1]
      3: vol_ratio               [0, ~10]
      4: momentum                [-0.1, 0.1]
      5: trade_count (norm)      [0, 1]
      6: time_to_expiry (log)    [0, ~4]
      7: price_range             [0, 1]
      8: bid_depth_3 (norm)      [0, 1]
      9: ask_depth_3 (norm)      [0, 1]
     10: book_imbalance          [-1, 1]
     11: btc_spot (log-norm)     [-0.1, 0.1]
     12: btc_return_5m           [-0.05, 0.05]
     13: btc_return_1h           [-0.1, 0.1]
     14: btc_funding_rate        [-0.01, 0.01]
     15: current_position (norm) [-1, 1]
     16: unrealized_pnl          [-1, 1]
     17: total_exposure (norm)   [0, 1]
    """
    # Market features
    spread = 0.0
    if orderbook.yes_price is not None and orderbook.no_price is not None:
        # Spread = ask - bid. For Kalshi: yes_ask ~ 1 - no_price, yes_bid ~ yes_price
        # Simpler: just measure the gap
        yes_bid = orderbook.yes_price if orderbook.yes_price is not None else 0.0
        no_bid = orderbook.no_price if orderbook.no_price is not None else 0.0
        # yes_ask ~ 1 - no_bid (approximately)
        spread = max(0.0, (1.0 - no_bid) - yes_bid) if no_bid > 0 else 0.0

    # Orderbook depth within 3 cents of best
    # Using the top-of-book sizes as proxy (full depth would require more data)
    bid_depth = min(float(orderbook.yes_size) / _DEPTH_NORM, 1.0)
    ask_depth = min(float(orderbook.no_size) / _DEPTH_NORM, 1.0)
    total_depth = bid_depth + ask_depth
    book_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

    # BTC spot features
    session_start = btc_poller.session_start_price()
    if session_start > 0:
        btc_log_norm = math.log(btc_poller.spot_price() / session_start) if btc_poller.spot_price() > 0 else 0.0
    else:
        btc_log_norm = 0.0

    # Position features
    position = pnl_tracker.get_position(ticker)
    position_norm = position / cfg.max_position_per_market
    unrealized = pnl_tracker.get_unrealized_pnl(ticker, market_price)
    exposure_norm = pnl_tracker.total_exposure() / cfg.max_total_markets

    state = np.array([
        market_price,                                        # 0
        spread,                                              # 1
        vol_metrics.get("vol", 0.0),                         # 2
        vol_metrics.get("vol_ratio", 0.0),                   # 3
        vol_metrics.get("momentum", 0.0),                    # 4
        min(trade_count / _TRADE_COUNT_NORM, 1.0),           # 5
        math.log(1.0 + max(0.0, time_to_expiry_h)),          # 6
        vol_metrics.get("range", 0.0),                       # 7
        bid_depth,                                           # 8
        ask_depth,                                           # 9
        book_imbalance,                                      # 10
        btc_log_norm,                                        # 11
        btc_poller.return_5m(),                              # 12
        btc_poller.return_1h(),                              # 13
        btc_poller.funding_rate(),                           # 14
        position_norm,                                       # 15
        unrealized,                                          # 16
        exposure_norm,                                       # 17
    ], dtype=np.float32)

    return state


def build_action_mask(
    ticker: str,
    pnl_tracker: PnLTracker,
    cfg: RLConfig,
) -> np.ndarray:
    """Build a binary mask of valid actions for a given market.

    Returns shape (n_actions,) with 1.0 for valid, 0.0 for invalid.

    Masking rules:
      - Position at +max: all BUY_YES (0-8) masked
      - Position at -max: all BUY_NO (9-17) masked
      - No YES position: CLOSE_YES masked
      - No NO position: CLOSE_NO masked
      - Exposure maxed and no position here: all BUY masked
    """
    mask = np.ones(cfg.n_actions, dtype=np.float32)
    position = pnl_tracker.get_position(ticker)

    # Can't close what you don't have
    if position <= 0:
        mask[ACTION_CLOSE_YES] = 0.0
    if position >= 0:
        mask[ACTION_CLOSE_NO] = 0.0

    # Position limits per market
    if position >= cfg.max_position_per_market:
        # Can't buy more YES
        for i in range(9):
            mask[i] = 0.0
    if position <= -cfg.max_position_per_market:
        # Can't buy more NO
        for i in range(9, 18):
            mask[i] = 0.0

    # Total exposure limit: if maxed and no position in this market, block all buys
    if position == 0 and pnl_tracker.total_exposure() >= cfg.max_total_markets:
        for i in range(18):
            mask[i] = 0.0

    return mask
