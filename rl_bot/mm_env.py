"""Market-making environment with multi-ticker support."""

import gymnasium
import numpy as np
import polars as pl
import math
from datetime import datetime, timezone
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader
from model.hp_dfm_rte.orderbook import OrderbookSnapshot
from rl_bot.reward import compute_maker_fee


def _ensure_datetime(trades_df: pl.DataFrame) -> pl.DataFrame:
    """Cast created_time to datetime if it's still a string."""
    if trades_df["created_time"].dtype == pl.Utf8:
        trades_df = trades_df.with_columns(
            pl.col("created_time")
            .str.replace(r"\+00:00$", "Z")
            .str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ", time_zone="UTC")
        )
    return trades_df


def preprocess_mm_data(
    trades_df: pl.DataFrame,
    orderbooks_df: pl.DataFrame | None = None,
    window_size_s: int = 60,
    split_date: str | None = None,
    split_mode: str = "train",
) -> dict[str, list[dict]]:
    """Merge trades and orderbooks into time-aligned windows.

    Vectorized implementation: uses Polars group_by instead of row-by-row
    Python iteration. Stores per-window trade data as numpy arrays for fast
    fill simulation (no dict hashing per trade in step()).

    Args:
        split_date: ISO date string (e.g. "2026-04-23") for temporal split.
                    None = no split, use all data.
        split_mode: "train" = data before split_date, "test" = data on/after split_date.
    """
    trades_df = _ensure_datetime(trades_df)
    trades_df = trades_df.sort(["ticker", "created_time"])

    # Apply temporal split filter
    if split_date is not None:
        if "T" in split_date:
            cutoff = datetime.strptime(split_date, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            cutoff = datetime.strptime(split_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if split_mode == "train":
            trades_df = trades_df.filter(pl.col("created_time") < cutoff)
        else:
            trades_df = trades_df.filter(pl.col("created_time") >= cutoff)

    # Truncate to window boundaries
    trades_df = trades_df.with_columns(
        pl.col("created_time").dt.truncate(f"{window_size_s}s").alias("window")
    )

    # Encode taker_side as int for numpy: 0 = "no" (buy), 1 = "yes" (sell)
    trades_df = trades_df.with_columns(
        (pl.col("taker_side") == "yes").cast(pl.Int8).alias("side_int")
    )

    # Pre-compute per-window aggregates via Polars group_by (no Python loops)
    agg_df = trades_df.group_by(["ticker", "window"]).agg([
        # VWAP numerator/denominator for mid-price fallback
        (pl.col("yes_price") * pl.col("count")).sum().alias("price_x_count"),
        pl.col("count").sum().alias("total_count"),
        # Trade count for volume observation
        pl.col("count").len().alias("n_trades"),
        # Raw arrays for fill simulation (stored as lists, converted to numpy per window)
        pl.col("yes_price").alias("prices"),
        pl.col("count").alias("counts"),
        pl.col("side_int").alias("sides"),
    ]).sort(["ticker", "window"])

    # Build result dict with numpy arrays per window
    result: dict[str, list[dict]] = {}
    # Partition by ticker — each partition is already sorted by window
    for ticker_df in agg_df.partition_by("ticker", maintain_order=True):
        ticker = ticker_df["ticker"][0]
        n_windows = len(ticker_df)
        windows: list[dict] = [None] * n_windows  # pre-allocate list

        # Extract columns as Python lists for fast iteration (no per-row Polars overhead)
        timestamps = ticker_df["window"].to_list()
        prices_col = ticker_df["prices"].to_list()
        counts_col = ticker_df["counts"].to_list()
        sides_col = ticker_df["sides"].to_list()
        vwap_num = ticker_df["price_x_count"].to_list()
        vwap_den = ticker_df["total_count"].to_list()
        n_trades_col = ticker_df["n_trades"].to_list()

        for i in range(n_windows):
            # Convert list columns to numpy arrays (contiguous, cache-friendly)
            windows[i] = {
                "prices": np.array(prices_col[i], dtype=np.float64),
                "counts": np.array(counts_col[i], dtype=np.int32),
                "sides": np.array(sides_col[i], dtype=np.int8),
                "vwap": vwap_num[i] / vwap_den[i] if vwap_den[i] > 0 else 0.5,
                "n_trades": n_trades_col[i],
                "total_count": vwap_den[i],
                "orderbook": None,
                "timestamp": timestamps[i],
            }

        result[ticker] = windows

    return result


def scale_action(action: np.ndarray, cfg: MMConfig) -> tuple[float, float]:
    """Convert action array to half_spread and skew values.

    Args:
        action: np.array([half_spread_control, skew_control]) in range [-1, 1]
        cfg: MMConfig with spread bounds

    Returns:
        (half_spread, skew) tuple
        - half_spread: in range [0.01, 0.10] (minimum 1 cent, maximum 10 cents)
        - skew: in range [-0.05, 0.05] (skew prices by up to 5 cents)
    """
    # Map half_spread_control from [-1, 1] to [0.01, 0.10]
    half_spread_control = np.clip(action[0], -1.0, 1.0)
    half_spread = 0.01 + (half_spread_control + 1.0) / 2.0 * 0.09  # [0.01, 0.10]

    # Map skew_control from [-1, 1] to [-0.05, 0.05]
    skew_control = np.clip(action[1], -1.0, 1.0)
    skew = skew_control * 0.05  # [-0.05, 0.05]

    return round(half_spread, 3), round(skew, 3)


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
        self._prev_value = 0.0  # previous step's total value for delta reward
        self._avg_entry_price = 0.5  # weighted avg entry price for position
        self._tte_hours = 24.0
        self._current_bid = 0.0
        self._current_ask = 0.0
        self._current_half_spread = 0.0
        self._current_skew = 0.0
        self._current_timestamp = None

        # Action space: [half_spread_control, skew_control]
        # Both in range [-1, 1]
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

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

        # Trade volume in current window (use pre-aggregated count)
        n_trades = window.get("n_trades", 0)
        volume_1m = min(n_trades / 50.0, 1.0) if n_trades else 0.0

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
        if self._inventory == 0:
            return 0.0
        return self._inventory * (self._mid - self._avg_entry_price)

    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state.

        Returns:
            observation, info
        """
        super().reset(seed=seed, options=options)
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
        self._prev_value = 0.0
        self._avg_entry_price = 0.5
        self._tte_hours = 24.0
        self._current_bid = 0.0
        self._current_ask = 0.0
        self._current_half_spread = 0.0
        self._current_skew = 0.0
        self._current_timestamp = None

        # Build initial observation
        obs = self._build_obs()
        info = {
            "ticker": self._current_ticker,
            "step": 0,
        }
        return obs, info

    def step(self, action: np.ndarray):
        """Execute one step of the environment with market-making action.

        Args:
            action: np.array([half_spread_control, skew_control]) in range [-1, 1]

        Returns:
            observation: 16-dim state vector
            reward: realized PnL change from fills
            terminated: True if episode is done (ran out of data)
            truncated: False (not used for now)
            info: dict with diagnostic information
        """
        # Check if episode is done (no more windows)
        if self._step_idx >= len(self._windows):
            terminated = True
            obs = self._build_obs()
            info = {
                "ticker": self._current_ticker,
                "timestamp": self._current_timestamp,
                "bid": self._current_bid,
                "ask": self._current_ask,
                "inventory": self._inventory,
                "fills_buy": self._fills_buy,
                "fills_sell": self._fills_sell,
                "pnl": self._realized_pnl + self._unrealized_pnl(),
                "reward": 0.0,
                "half_spread": self._current_half_spread,
                "skew": self._current_skew,
                "realized_pnl": self._realized_pnl,
                "unrealized_pnl": self._unrealized_pnl(),
                "mid_price": self._mid,
            }
            return obs, 0.0, terminated, False, info

        # Get current window
        window = self._windows[self._step_idx]
        self._current_timestamp = window.get("timestamp")

        # Update mid price from orderbook or pre-computed VWAP
        orderbook = window.get("orderbook")
        if orderbook is not None and orderbook.mid_price() is not None:
            self._mid = orderbook.mid_price()
        else:
            # Use pre-aggregated VWAP (computed in preprocess_mm_data)
            vwap = window.get("vwap")
            if vwap is not None and vwap > 0:
                self._mid = vwap

        # Round mid to 3 decimals
        self._mid = round(self._mid, 3)
        self._mid_history.append(self._mid)

        # Compute bid/ask from mid + action
        half_spread, skew = scale_action(action, self._cfg)
        self._current_half_spread = half_spread
        self._current_skew = skew

        # Base bid/ask (before subpenny adjustment)
        bid_base = self._mid - half_spread + skew
        ask_base = self._mid + half_spread + skew

        # Apply subpenny adjustment for queue priority (task 4)
        bid = self._apply_subpenny_adjustment(bid_base, "bid")
        ask = self._apply_subpenny_adjustment(ask_base, "ask")

        # Clamp to valid Kalshi range [0.01, 0.99]
        bid = max(0.01, min(0.99, bid))
        ask = max(0.01, min(0.99, ask))

        # Ensure ask > bid (prevent crossing)
        if ask <= bid:
            # If adjustment caused crossing, revert to base prices
            bid = max(0.01, min(0.99, bid_base))
            ask = max(0.01, min(0.99, ask_base))

        # Round to 3 decimals
        bid = round(bid, 3)
        ask = round(ask, 3)

        self._current_bid = bid
        self._current_ask = ask

        # Simulate fills using numpy vectorized ops (no Python per-trade loop)
        # Queue priority: subpenny orders jump ahead of makers at rounded cent
        has_subpenny_bid = (bid != bid_base) and self._cfg.subpenny_enabled
        has_subpenny_ask = (ask != ask_base) and self._cfg.subpenny_enabled
        queue_competitors = 5  # assumed number of other makers at same cent level

        prices = window.get("prices")
        if prices is not None and len(prices) > 0:
            counts = window["counts"]
            sides = window["sides"]

            # sides: 0 = "no" (taker bought NO -> we buy YES), 1 = "yes" (taker bought YES -> we sell YES)
            # Vectorized mask for buy fills: taker_side=="no" AND price <= bid
            buy_mask = (sides == 0) & (prices <= bid)
            # Vectorized mask for sell fills: taker_side=="yes" AND price >= ask
            sell_mask = (sides == 1) & (prices >= ask)

            # Process buy fills
            if buy_mask.any():
                buy_prices = prices[buy_mask]
                buy_counts = counts[buy_mask]
                if not has_subpenny_bid:
                    buy_counts = np.maximum(1, buy_counts // queue_competitors)
                for j in range(len(buy_prices)):
                    self._fill_buy(float(buy_prices[j]), int(buy_counts[j]))

            # Process sell fills
            if sell_mask.any():
                sell_prices = prices[sell_mask]
                sell_counts = counts[sell_mask]
                if not has_subpenny_ask:
                    sell_counts = np.maximum(1, sell_counts // queue_competitors)
                for j in range(len(sell_prices)):
                    self._fill_sell(float(sell_prices[j]), int(sell_counts[j]))

        # Compute reward = delta(realized + unrealized PnL) - inventory penalty
        new_value = self._realized_pnl + self._unrealized_pnl()
        delta_pnl = new_value - self._prev_value
        self._prev_value = new_value

        # Inventory penalty (scaled by time-to-expiry if enabled)
        inv_penalty = self._cfg.inventory_lambda * abs(self._inventory)
        if self._cfg.inventory_tte_scale:
            inv_penalty *= 1.0 / math.sqrt(self._tte_hours + 1.0)

        reward = delta_pnl - inv_penalty

        # Advance step and decrement time-to-expiry
        self._step_idx += 1
        self._tte_hours = max(0.0, self._tte_hours - (1.0 / 60.0))

        # Check if episode is done
        done = self._step_idx >= len(self._windows)

        # Flatten inventory at episode end (realize at last mid, no exit fee)
        if done and self._inventory != 0:
            flatten_pnl = self._inventory * (self._mid - self._avg_entry_price)
            self._realized_pnl += flatten_pnl
            self._inventory = 0

        # Build next observation
        obs = self._build_obs()

        # Build info dict
        info = {
            "ticker": self._current_ticker,
            "timestamp": self._current_timestamp,
            "bid": self._current_bid,
            "ask": self._current_ask,
            "inventory": self._inventory,
            "fills_buy": self._fills_buy,
            "fills_sell": self._fills_sell,
            "pnl": self._realized_pnl + self._unrealized_pnl(),
            "reward": reward,
            "half_spread": self._current_half_spread,
            "skew": self._current_skew,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl(),
            "mid_price": self._mid,
        }

        return obs, reward, done, False, info

    def _fill_buy(self, price: float, size: int) -> None:
        """Execute a buy fill (agent buys YES contracts).
        Respects inventory cap. Deducts maker fee from realized PnL.
        """
        # Cap size to stay within max_inventory
        if self._inventory + size > self._cfg.max_inventory:
            size = max(0, self._cfg.max_inventory - self._inventory)
        if size <= 0:
            return

        # Deduct maker fee
        fee = compute_maker_fee(size, price, self._cfg.maker_fee_rate)
        self._realized_pnl -= fee

        # Update weighted avg entry price
        old_inv = self._inventory
        self._inventory += size
        if self._inventory > 0:
            self._avg_entry_price = (old_inv * self._avg_entry_price + size * price) / self._inventory
        self._fills_buy += size

    def _fill_sell(self, price: float, size: int) -> None:
        """Execute a sell fill (agent sells YES contracts).
        Respects inventory cap (can't go more short than -max_inventory).
        """
        # Cap size to stay within -max_inventory
        if self._inventory - size < -self._cfg.max_inventory:
            size = max(0, self._inventory + self._cfg.max_inventory)
        if size <= 0:
            return

        # Deduct maker fee
        fee = compute_maker_fee(size, price, self._cfg.maker_fee_rate)
        self._realized_pnl -= fee

        # Update position
        old_inv = self._inventory
        self._inventory -= size
        # Update avg entry price on direction flip
        if self._inventory != 0 and ((old_inv > 0 and self._inventory < 0) or (old_inv < 0 and self._inventory > 0)):
            self._avg_entry_price = price
        elif self._inventory == 0:
            self._avg_entry_price = self._mid
        self._fills_sell += size

    def _apply_subpenny_adjustment(self, price: float, side: str) -> float:
        """Apply subpenny adjustment for queue priority if market supports it.

        Args:
            price: Base price computed from mid + spread + skew
            side: "bid" or "ask"

        Returns:
            Adjusted price (or original if subpenny not valid)
        """
        # Check if subpenny feature is enabled
        if not self._cfg.subpenny_enabled:
            return price

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
