"""Market-Making Gymnasium Environment for PPO agent training.

Simulates market-making on historical Kalshi trade data. Cycles through multiple
tickers to learn a market-agnostic policy. Fill simulation uses trade-by-trade
matching against posted bid/ask quotes.

Data format:
  - Input: Polars DataFrame with columns: ticker, yes_price, count, taker_side, created_time
  - Preprocessing: groups by ticker, then into 1-minute windows
  - Each window is a list of trade dicts for that minute

Key design choices:
  - O(1) per-step, O(n) total complexity
  - Multi-ticker support from the start (cycles on reset)
  - Inventory cap enforced on all fills
  - Maker fees deducted via compute_maker_fee()
  - Episode ends when all windows consumed; flatten inventory at last mid (no exit fee)
  - Mid price = VWAP of current window's trades; uses previous mid if window empty
  - Time-to-expiry: default 24h, decrements 1min per step (sufficient for v1)
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium
import numpy as np
import polars as pl

from rl_bot.execution import estimate_spread
from rl_bot.mm_config import MMConfig, scale_action
from rl_bot.reward import compute_maker_fee


def preprocess_trades_for_mm(df: pl.DataFrame) -> dict[str, list[list[dict]]]:
    """Group trades into per-ticker, per-minute windows.

    Each ticker gets a list of windows (1-minute intervals). Each window is a
    list of trade dicts with fields: yes_price, count, taker_side, created_time.

    Args:
        df: Polars DataFrame with columns: ticker, yes_price, count, taker_side, created_time

    Returns:
        dict mapping ticker -> list of windows (each window is a list of trade dicts)

    Time complexity: O(n) where n = number of trades
    """
    # Sort by ticker and time for efficient grouping
    df = df.sort(["ticker", "created_time"])

    # Convert created_time to datetime if it's a string
    if df["created_time"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("created_time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ")
        )

    # Add 1-minute window column (floor to nearest minute)
    df = df.with_columns(
        pl.col("created_time").dt.truncate("1m").alias("window")
    )

    # Group by ticker and window, collect trades into lists
    result: dict[str, list[list[dict]]] = {}

    for ticker in df["ticker"].unique().to_list():
        ticker_df = df.filter(pl.col("ticker") == ticker)
        windows: list[list[dict]] = []

        for window_time in ticker_df["window"].unique().sort().to_list():
            window_df = ticker_df.filter(pl.col("window") == window_time)
            # Convert to list of dicts, keep only needed fields
            trades = []
            for row in window_df.iter_rows(named=True):
                trades.append({
                    "yes_price": float(row["yes_price"]),
                    "count": int(row["count"]),
                    "taker_side": row["taker_side"],
                    "created_time": row["created_time"],
                })
            windows.append(trades)

        result[ticker] = windows

    return result


class MMEnv(gymnasium.Env):
    """Market-making environment with multi-ticker support.

    Observation space: Box(10,) with market-agnostic features
    Action space: Box([-1, -1], [1, 1]) -> (half_spread, skew) via scale_action()

    Step logic:
      1. Scale action to (half_spread, skew)
      2. Compute bid/ask from mid price
      3. Simulate fills for all trades in current 1-minute window
      4. Apply inventory cap and maker fees
      5. Compute reward = delta_pnl - inventory_penalty
      6. Return obs, reward, done, truncated, info

    Multi-ticker cycling:
      - __init__ stores all ticker data
      - reset() cycles to next ticker (round-robin)
      - Agent learns general MM policy across all markets
    """

    metadata = {"render_modes": []}

    def __init__(self, ticker_data: dict[str, list[list[dict]]], cfg: MMConfig):
        """Initialize environment with preprocessed trade data.

        Args:
            ticker_data: output of preprocess_trades_for_mm()
            cfg: MMConfig instance with all hyperparameters
        """
        super().__init__()

        self._cfg = cfg
        self._ticker_data = ticker_data
        self._tickers = list(ticker_data.keys())
        self._ticker_idx = -1  # Will cycle to 0 on first reset

        # Gymnasium spaces
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0.0, 0.01, 0.0, 0.0, -1.0, -1.0, 0.0, -0.1, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 0.10, 1.0, 1.0, 1.0, 1.0, 4.0, 0.1, 1.0, 1.0], dtype=np.float32),
            shape=(10,),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode state (initialized in reset())
        self._windows: list[list[dict]] = []
        self._step_idx = 0
        self._inventory = 0
        self._realized_pnl = 0.0
        self._mid = 0.50  # Default mid price
        self._prev_mid = 0.50
        self._tte_hours = 24.0  # Default time-to-expiry
        self._prev_value = 0.0
        self._inventory_flat_step = 0  # Last step when inventory was 0
        self._mid_history: list[float] = []  # For price momentum (last 5 steps)

        # Per-step state for logging
        self._current_ticker = ""
        self._fills_buy = 0  # Contracts bought this step
        self._fills_sell = 0  # Contracts sold this step
        self._current_bid = 0.0
        self._current_ask = 0.0
        self._current_half_spread = 0.0
        self._current_skew = 0.0
        self._current_timestamp = ""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to next ticker (round-robin), clear inventory/PnL.

        Returns:
            tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        # Cycle to next ticker
        self._ticker_idx = (self._ticker_idx + 1) % len(self._tickers)
        ticker = self._tickers[self._ticker_idx]
        self._windows = self._ticker_data[ticker]
        self._current_ticker = ticker

        # Reset episode state
        self._step_idx = 0
        self._inventory = 0
        self._realized_pnl = 0.0
        self._tte_hours = 24.0
        self._prev_value = 0.0
        self._inventory_flat_step = 0
        self._mid_history = []

        # Reset per-step state
        self._fills_buy = 0
        self._fills_sell = 0
        self._current_bid = 0.0
        self._current_ask = 0.0
        self._current_half_spread = 0.0
        self._current_skew = 0.0
        self._current_timestamp = ""

        # Compute initial mid price from first window
        if self._windows and self._windows[0]:
            self._mid = self._compute_vwap(self._windows[0])
        else:
            self._mid = 0.50
        self._prev_mid = self._mid
        self._mid_history.append(self._mid)

        obs = self._build_obs()
        info = {"ticker": ticker}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step of market-making.

        Args:
            action: 2D array with values in [-1, 1] for (half_spread, skew)

        Returns:
            tuple of (obs, reward, terminated, truncated, info)
        """
        # Reset per-step counters
        self._fills_buy = 0
        self._fills_sell = 0

        # Scale action to actual quote parameters
        half_spread, skew = scale_action(action, self._cfg)
        self._current_half_spread = half_spread
        self._current_skew = skew

        # Current window's trades
        current_window = self._windows[self._step_idx]

        # Compute mid price for this window (VWAP if trades exist, else previous mid)
        if current_window:
            self._mid = self._compute_vwap(current_window)
            # Store timestamp from first trade in window
            self._current_timestamp = str(current_window[0]["created_time"])
        # else: keep previous mid

        # Compute bid/ask from mid + action
        bid = self._mid - half_spread + skew
        ask = self._mid + half_spread + skew

        # Clamp to valid Kalshi range [0.01, 0.99]
        bid = max(0.01, min(0.99, bid))
        ask = max(0.01, min(0.99, ask))

        self._current_bid = bid
        self._current_ask = ask

        # Simulate fills for all trades in this window
        for trade in current_window:
            trade_price = trade["yes_price"]
            trade_size = trade["count"]
            taker_side = trade["taker_side"]

            # Fill logic: taker_side indicates which side the taker bought
            # If taker bought NO (taker_side="no"), they're hitting bids (we buy YES)
            # If taker bought YES (taker_side="yes"), they're hitting asks (we sell YES)
            if taker_side == "no" and trade_price <= bid:
                self._fill_buy(trade_price, trade_size)
            elif taker_side == "yes" and trade_price >= ask:
                self._fill_sell(trade_price, trade_size)

        # Update mid history for momentum calculation
        self._mid_history.append(self._mid)
        if len(self._mid_history) > 5:
            self._mid_history.pop(0)

        # Compute reward
        new_value = self._realized_pnl + self._unrealized_pnl()
        delta_pnl = new_value - self._prev_value
        self._prev_value = new_value

        # Inventory penalty (scaled by time-to-expiry if enabled)
        inv_penalty = self._cfg.inventory_lambda * abs(self._inventory)
        if self._cfg.inventory_tte_scale:
            inv_penalty *= 1.0 / math.sqrt(self._tte_hours + 1.0)

        reward = delta_pnl - inv_penalty

        # Advance step
        self._step_idx += 1
        self._tte_hours = max(0.0, self._tte_hours - (1.0 / 60.0))  # Decrement by 1 minute

        # Check if episode is done
        done = self._step_idx >= len(self._windows)

        # If episode ends, flatten inventory at last mid price (no exit fee)
        if done and self._inventory != 0:
            # Realize PnL from flattening inventory
            flatten_pnl = self._inventory * (self._mid - self._get_avg_entry_price())
            self._realized_pnl += flatten_pnl
            self._inventory = 0

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
            "reward": reward,
            "half_spread": self._current_half_spread,
            "skew": self._current_skew,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl(),
            "mid_price": self._mid,
        }

        return obs, reward, done, done, info

    def _compute_vwap(self, trades: list[dict]) -> float:
        """Compute volume-weighted average price for a list of trades.

        Args:
            trades: list of trade dicts with yes_price and count fields

        Returns:
            VWAP (clamped to [0.01, 0.99])
        """
        if not trades:
            return self._mid  # Fallback to previous mid

        total_value = 0.0
        total_volume = 0.0
        for t in trades:
            total_value += t["yes_price"] * t["count"]
            total_volume += t["count"]

        if total_volume > 0:
            vwap = total_value / total_volume
            return max(0.01, min(0.99, vwap))
        return self._mid

    def _fill_buy(self, price: float, size: int) -> None:
        """Execute a buy fill (agent buys YES contracts).

        Respects inventory cap. Deducts maker fee from realized PnL.
        Updates inventory and entry price tracking.

        Args:
            price: YES-side fill price
            size: number of contracts
        """
        # Apply inventory cap
        if self._inventory + size > self._cfg.max_inventory:
            # Fill only what fits under cap
            size = max(0, self._cfg.max_inventory - self._inventory)

        if size <= 0:
            return

        # Track fills for this step
        self._fills_buy += size

        # Deduct maker fee
        fee = compute_maker_fee(size, price, self._cfg.maker_fee_rate)
        self._realized_pnl -= fee

        # Update position (for FIFO, we simplify to avg entry price)
        # In v1, we track net inventory and weighted avg entry price
        old_inv = self._inventory
        old_entry = self._get_avg_entry_price()

        self._inventory += size
        # Weighted average entry price
        new_entry = (old_inv * old_entry + size * price) / self._inventory if self._inventory > 0 else price
        self._avg_entry_price = new_entry

        # Track when inventory was last flat
        if old_inv == 0:
            self._inventory_flat_step = self._step_idx

    def _fill_sell(self, price: float, size: int) -> None:
        """Execute a sell fill (agent sells YES contracts).

        Respects inventory cap (can't go more short than -max_inventory).
        Deducts maker fee from realized PnL. Updates inventory and entry price.

        Args:
            price: YES-side fill price
            size: number of contracts
        """
        # Apply inventory cap (negative direction)
        if self._inventory - size < -self._cfg.max_inventory:
            # Fill only what fits under cap
            size = max(0, self._inventory + self._cfg.max_inventory)

        if size <= 0:
            return

        # Track fills for this step
        self._fills_sell += size

        # Deduct maker fee
        fee = compute_maker_fee(size, price, self._cfg.maker_fee_rate)
        self._realized_pnl -= fee

        # Update position
        old_inv = self._inventory
        old_entry = self._get_avg_entry_price()

        self._inventory -= size
        # Weighted average entry price
        if self._inventory != 0:
            # Reduce position or flip direction
            if old_inv > 0 and self._inventory >= 0:
                # Reducing long position (keep same avg entry)
                pass
            elif old_inv < 0 and self._inventory <= 0:
                # Reducing short position (keep same avg entry)
                pass
            else:
                # Flipping from long to short or vice versa
                new_entry = (old_inv * old_entry - size * price) / self._inventory if self._inventory != 0 else price
                self._avg_entry_price = new_entry
        else:
            # Went flat
            self._avg_entry_price = self._mid
            self._inventory_flat_step = self._step_idx

    def _get_avg_entry_price(self) -> float:
        """Return current average entry price (or mid if flat).

        Returns:
            Average entry price for current position
        """
        if not hasattr(self, "_avg_entry_price") or self._inventory == 0:
            return self._mid
        return self._avg_entry_price

    def _unrealized_pnl(self) -> float:
        """Compute mark-to-market unrealized PnL based on current mid price.

        Returns:
            Unrealized PnL (positive = profit, negative = loss)
        """
        if self._inventory == 0:
            return 0.0

        entry = self._get_avg_entry_price()
        return self._inventory * (self._mid - entry)

    def _build_obs(self) -> np.ndarray:
        """Build observation vector (10 features, all market-agnostic).

        Returns:
            numpy array of shape (10,) with dtype float32
        """
        # Current window for volume/ratio calculation
        current_window = self._windows[self._step_idx] if self._step_idx < len(self._windows) else []

        # Feature 0: mid_price [0, 1]
        mid_price_norm = float(self._mid)

        # Feature 1: spread_est [0.01, 0.10]
        # Use estimate_spread with current market conditions
        trade_volume = len(current_window)  # Approximation: number of trades as proxy
        spread_est = estimate_spread(self._mid, self._tte_hours, trade_volume)

        # Feature 2: trade_volume [0, 1] (normalized count in window)
        # Normalize by typical max (~100 trades per minute is high activity)
        trade_volume_norm = min(1.0, len(current_window) / 100.0)

        # Feature 3: buy_ratio [0, 1] (fraction taker_side="yes")
        buy_count = sum(1 for t in current_window if t["taker_side"] == "yes")
        buy_ratio = buy_count / len(current_window) if current_window else 0.5

        # Feature 4: inventory_norm [-1, 1]
        inventory_norm = self._inventory / self._cfg.max_inventory

        # Feature 5: unrealized_pnl [-1, 1] (clamped)
        # Normalize by typical max (~$20 for 20 contracts at $1 move)
        unrealized = self._unrealized_pnl()
        unrealized_norm = np.clip(unrealized / 20.0, -1.0, 1.0)

        # Feature 6: time_to_expiry [0, 4] (log scale)
        tte_norm = math.log(1.0 + self._tte_hours)

        # Feature 7: price_momentum [-0.1, 0.1] (5-min price change)
        if len(self._mid_history) >= 2:
            # Compare current to 5 steps ago (5 minutes)
            lookback_idx = max(0, len(self._mid_history) - 6)
            price_momentum = self._mid - self._mid_history[lookback_idx]
            price_momentum = np.clip(price_momentum, -0.1, 0.1)
        else:
            price_momentum = 0.0

        # Feature 8: realized_pnl_norm [-1, 1] (session PnL)
        # Normalize by typical max (~$50 for a good session)
        realized_norm = np.clip(self._realized_pnl / 50.0, -1.0, 1.0)

        # Feature 9: inventory_age [0, 1] (minutes since flat / 60)
        minutes_since_flat = self._step_idx - self._inventory_flat_step
        inventory_age = min(1.0, minutes_since_flat / 60.0)

        obs = np.array(
            [
                mid_price_norm,
                spread_est,
                trade_volume_norm,
                buy_ratio,
                inventory_norm,
                unrealized_norm,
                tte_norm,
                price_momentum,
                realized_norm,
                inventory_age,
            ],
            dtype=np.float32,
        )

        return obs
