"""Market-making environment with multi-ticker support."""

import gymnasium
import numpy as np
import polars as pl
import math
from datetime import datetime, timezone
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader
from model.hp_dfm_rte.orderbook import OrderbookSnapshot


def preprocess_mm_data(
    trades_df: pl.DataFrame,
    orderbooks_df: pl.DataFrame | None = None,
    window_size_s: int = 60,
) -> dict[str, list[dict]]:
    """Merge trades and orderbooks into time-aligned windows.

    TODO: Full implementation in Task 6
    """
    # Minimal implementation for now
    trades_df = trades_df.sort(["ticker", "created_time"])

    if trades_df["created_time"].dtype == pl.Utf8:
        trades_df = trades_df.with_columns(
            pl.col("created_time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ")
        )

    trades_df = trades_df.with_columns(
        pl.col("created_time").dt.truncate(f"{window_size_s}s").alias("window")
    )

    result: dict[str, list[dict]] = {}

    for ticker in trades_df["ticker"].unique().sort().to_list():
        ticker_trades = trades_df.filter(pl.col("ticker") == ticker)
        windows: list[dict] = []

        for window_time in ticker_trades["window"].unique().sort().to_list():
            window_trades_df = ticker_trades.filter(pl.col("window") == window_time)
            trades_list = []
            for row in window_trades_df.iter_rows(named=True):
                trades_list.append({
                    "yes_price": float(row["yes_price"]),
                    "count": int(row["count"]),
                    "taker_side": row["taker_side"],
                    "created_time": row["created_time"],
                })

            windows.append({
                "trades": trades_list,
                "orderbook": None,  # Will be added in Task 6
                "timestamp": window_time,
            })

        result[ticker] = windows

    return result


class MMEnv(gymnasium.Env):
    """Market-making environment with multi-ticker support."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ticker_data: dict[str, list[dict]],
        cfg: MMConfig,
        metadata_loader: MarketMetadataLoader | None = None,
    ):
        """Initialize environment with preprocessed trade data.

        Args:
            ticker_data: output of preprocess_mm_data()
            cfg: MMConfig instance with all hyperparameters
            metadata_loader: Market metadata loader for tick validation
        """
        super().__init__()

        self._cfg = cfg
        self._ticker_data = ticker_data
        self._tickers = list(ticker_data.keys())
        self._ticker_idx = -1
        self._metadata_loader = metadata_loader
        self._current_ticker = None

        # Gymnasium spaces - 16 dimensions
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([
                0.01,   # [0] mid_price
                0.01,   # [1] spread
                0.0,    # [2] bid_depth_l0
                0.0,    # [3] ask_depth_l0
                0.0,    # [4] bid_depth_l1
                0.0,    # [5] ask_depth_l1
                0.0,    # [6] bid_depth_l2
                0.0,    # [7] ask_depth_l2
                -1.0,   # [8] book_imbalance
                0.0,    # [9] trade_volume_1m
                -1.0,   # [10] inventory_norm
                -1.0,   # [11] unrealized_pnl_norm
                0.0,    # [12] tte_log
                -0.1,   # [13] momentum
                -1.0,   # [14] realized_pnl_norm
                0.0,    # [15] fills_ratio
            ], dtype=np.float32),
            high=np.array([
                0.99,   # [0] mid_price
                0.10,   # [1] spread
                1.0,    # [2] bid_depth_l0
                1.0,    # [3] ask_depth_l0
                1.0,    # [4] bid_depth_l1
                1.0,    # [5] ask_depth_l1
                1.0,    # [6] bid_depth_l2
                1.0,    # [7] ask_depth_l2
                1.0,    # [8] book_imbalance
                1.0,    # [9] trade_volume_1m
                1.0,    # [10] inventory_norm
                1.0,    # [11] unrealized_pnl_norm
                4.0,    # [12] tte_log
                0.1,    # [13] momentum
                1.0,    # [14] realized_pnl_norm
                1.0,    # [15] fills_ratio
            ], dtype=np.float32),
            shape=(16,),
            dtype=np.float32,
        )

        # Initialize state tracking
        self._step_idx = 0
        self._windows = []
        self._mid = 0.5
        self._mid_history = []
        self._inventory = 0
        self._fills_buy = 0
        self._fills_sell = 0
        self._realized_pnl = 0.0
        self._tte_hours = 1.0

    def _build_obs(self) -> np.ndarray:
        """Build 16-dimensional observation vector.

        Returns:
            np.array with 16 elements representing market and position state
        """
        # Get current window's orderbook (may be None)
        window = self._windows[self._step_idx] if self._step_idx < len(self._windows) else {}
        orderbook = window.get("orderbook")

        # Orderbook features (fallback to defaults if missing)
        if orderbook is not None:
            spread = orderbook.spread() or 0.03
            bid_l0 = min(orderbook.yes_size / 100.0, 1.0)
            ask_l0 = min(orderbook.no_size / 100.0, 1.0)
            bid_l1 = min(orderbook.yes_size_l1 / 100.0, 1.0) if orderbook.yes_size_l1 else 0.05
            ask_l1 = min(orderbook.no_size_l1 / 100.0, 1.0) if orderbook.no_size_l1 else 0.05
            bid_l2 = min(orderbook.yes_size_l2 / 100.0, 1.0) if orderbook.yes_size_l2 else 0.02
            ask_l2 = min(orderbook.no_size_l2 / 100.0, 1.0) if orderbook.no_size_l2 else 0.02
            imbalance = orderbook.imbalance()
        else:
            # Fallback when no orderbook
            spread = 0.03
            bid_l0 = ask_l0 = 0.1
            bid_l1 = ask_l1 = 0.05
            bid_l2 = ask_l2 = 0.02
            imbalance = 0.0

        # Trade volume in current window
        trades = window.get("trades", [])
        volume_1m = min(len(trades) / 50.0, 1.0) if trades else 0.0

        # Position features
        inv_norm = self._inventory / max(self._cfg.max_inventory, 1)
        unrealized_norm = self._unrealized_pnl() / max(self._cfg.max_inventory * 0.5, 1)
        realized_norm = self._realized_pnl / 50.0

        # Time feature
        tte_log = math.log(1.0 + self._tte_hours)

        # Momentum (5-step lookback)
        if len(self._mid_history) >= 5:
            momentum = self._mid - self._mid_history[-5]
        else:
            momentum = 0.0

        # Fill ratio (how often our quotes got hit)
        fills_ratio = (self._fills_buy + self._fills_sell) / max(self._cfg.quote_size, 1.0)

        obs = np.array([
            self._mid,          # [0] mid_price
            spread,             # [1] spread
            bid_l0,             # [2] bid_depth_l0
            ask_l0,             # [3] ask_depth_l0
            bid_l1,             # [4] bid_depth_l1
            ask_l1,             # [5] ask_depth_l1
            bid_l2,             # [6] bid_depth_l2
            ask_l2,             # [7] ask_depth_l2
            imbalance,          # [8] book_imbalance
            volume_1m,          # [9] trade_volume_1m
            inv_norm,           # [10] inventory_norm
            unrealized_norm,    # [11] unrealized_pnl_norm
            tte_log,            # [12] tte_log
            momentum,           # [13] momentum
            realized_norm,      # [14] realized_pnl_norm
            fills_ratio,        # [15] fills_ratio
        ], dtype=np.float32)

        # Clip to valid ranges (safety)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _unrealized_pnl(self) -> float:
        """Calculate unrealized PnL based on current inventory and mid price."""
        return self._inventory * (self._mid - 0.5)

    def reset(self):
        """Reset environment to initial state.

        Returns:
            observation, info
        """
        if not self._tickers:
            # No data to reset to
            obs = np.zeros(16, dtype=np.float32)
            return obs, {}

        # Select next ticker (cycle through tickers)
        self._ticker_idx = (self._ticker_idx + 1) % len(self._tickers)
        self._current_ticker = self._tickers[self._ticker_idx]

        # Load windows for this ticker
        self._windows = self._ticker_data.get(self._current_ticker, [])

        # Reset state
        self._step_idx = 0
        self._mid = 0.5
        self._mid_history = []
        self._inventory = 0
        self._fills_buy = 0
        self._fills_sell = 0
        self._realized_pnl = 0.0
        self._tte_hours = 1.0

        # Build initial observation
        obs = self._build_obs()
        info = {
            "ticker": self._current_ticker,
            "step": 0,
        }
        return obs, info

    def _apply_subpenny_adjustment(self, price: float, side: str) -> float:
        """Apply subpenny adjustment for queue priority if market supports it.

        Args:
            price: Base price computed from mid + spread + skew
            side: "bid" or "ask"

        Returns:
            Adjusted price (or original if subpenny not valid)
        """
        # Check if metadata loader available
        if self._metadata_loader is None:
            return price  # No validation possible, return original

        # Check if current market supports subpenny at this price
        if not self._metadata_loader.supports_subpenny(self._current_ticker, price):
            return price  # No adjustment if tick size is 0.01

        # Apply queue-jumping adjustment
        if side == "bid":
            # Bid: add 0.001 to jump ahead (pay more)
            adjusted = price + 0.001
        else:  # side == "ask"
            # Ask: subtract 0.001 to jump ahead (sell cheaper)
            adjusted = price - 0.001

        # Ensure we stay within valid Kalshi range [0.01, 0.99]
        adjusted = max(0.01, min(0.99, adjusted))

        # Round to valid tick (0.001 precision)
        adjusted = round(adjusted, 3)

        return adjusted
