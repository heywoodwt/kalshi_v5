"""Trade stream -> bucketed price panel (rolling window).

Pipeline:
  1. Truncate trade timestamps to configurable-interval buckets (default 60s)
  2. Pivot to wide format (VWAP yes_price per bucket per ticker)
  3. Forward-fill, dropna
  4. Filter tickers with std < min_std or obs < min_observations
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from .config import PipelineConfig


class TradeAccumulator:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        # Per-ticker rolling deque of (bucket_key, price) tuples
        self._prices: dict[str, deque[tuple[str, float]]] = defaultdict(
            lambda: deque(maxlen=cfg.rolling_window)
        )
        # Current open bucket: {ticker: [(price, count)]} for VWAP computation
        self._current_bucket: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._current_bucket_key: str | None = None

    def _bucket_key(self, ts: datetime) -> str:
        """Truncate a datetime to the start of its bucket interval (ISO str)."""
        epoch = int(ts.timestamp())
        floored = epoch - (epoch % self.cfg.bucket_interval_s)
        return datetime.fromtimestamp(floored, tz=timezone.utc).isoformat()

    def ingest_trade(self, trade: dict[str, Any]) -> bool:
        """Ingest a single trade dict. Returns True when a new minute bucket closes.

        Expected trade keys:
            - market_ticker: str
            - yes_price: float
            - created_time: str (ISO 8601) or datetime
        """
        ticker: str = trade["market_ticker"]
        # WS uses "yes_price_dollars" (str); CSV uses "yes_price" (float)
        raw_price = (
            trade.get("yes_price")
            or trade.get("yes_price_dollars")
            or trade.get("price")
        )
        if raw_price is None:
            return False
        price: float = float(raw_price)
        count: int = max(1, int(trade.get("count", 1) or 1))

        # WS trades may use "created_time", "ts", or "traded_timestamp"
        ts = trade.get("created_time") or trade.get("traded_timestamp") or trade.get("ts")
        if ts is None:
            return False
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        bk = self._bucket_key(ts)

        bucket_closed = False
        if self._current_bucket_key is not None and bk != self._current_bucket_key:
            # Close the previous bucket
            self._close_bucket()
            bucket_closed = True

        self._current_bucket_key = bk
        self._current_bucket[ticker].append((price, count))
        return bucket_closed

    def _close_bucket(self) -> None:
        """Flush current bucket as volume-weighted average price (VWAP) into rolling deques."""
        bk = self._current_bucket_key
        for ticker, price_count_pairs in self._current_bucket.items():
            total_count = sum(c for _, c in price_count_pairs)
            if total_count > 0:
                vwap = sum(p * c for p, c in price_count_pairs) / total_count
            else:
                vwap = float(np.mean([p for p, _ in price_count_pairs]))
            self._prices[ticker].append((bk, vwap))
        self._current_bucket.clear()

    def get_price_panel(self) -> pl.DataFrame | None:
        """Build a wide DataFrame from rolling deques.

        Returns None if no valid tickers remain after filtering.
        """
        if not self._prices:
            return None

        # Step 1: Pre-filter tickers by raw observation count
        candidate_tickers = [
            ticker
            for ticker, dq in self._prices.items()
            if len(dq) >= self.cfg.min_observations
        ]
        if not candidate_tickers:
            return None

        # Step 2: Build panel from candidates only
        all_keys: set[str] = set()
        for ticker in candidate_tickers:
            for bk, _ in self._prices[ticker]:
                all_keys.add(bk)
        if not all_keys:
            return None

        index = sorted(all_keys)
        index_lookup = {k: i for i, k in enumerate(index)}

        # Build columnar data: one list per ticker, aligned to sorted index
        data: dict[str, list[float | None]] = {}
        for ticker in candidate_tickers:
            col = [None] * len(index)
            for bk, price in self._prices[ticker]:
                col[index_lookup[bk]] = price
            data[ticker] = col

        panel = pl.DataFrame({"time_bucket": index, **data})

        # Forward-fill nulls, then drop rows that still have any null
        panel = panel.with_columns(
            pl.col(c).forward_fill() for c in candidate_tickers
        ).drop_nulls()

        if panel.is_empty():
            return None

        # Step 3: Filter by std (remove near-constant tickers)
        valid_cols = [
            col
            for col in candidate_tickers
            if panel.get_column(col).std() > self.cfg.min_std_threshold
        ]
        if not valid_cols:
            return None

        # Step 4: Filter by unique price count -- stale/illiquid tickers
        valid_cols = [
            col
            for col in valid_cols
            if panel.get_column(col).n_unique() >= self.cfg.min_unique_prices
        ]
        if not valid_cols:
            return None

        return panel.select(["time_bucket"] + valid_cols)