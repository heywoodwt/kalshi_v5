"""Market metadata loader for subpenny pricing validation.

Supports dual-mode operation:
- Parquet mode (training): Load from markets.parquet file
- API mode (production): Fetch from Kalshi REST API with caching
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class MarketMetadata:
    """Metadata for a single market."""
    ticker: str
    price_level_structure: str  # "linear_cent" | "tapered_deci_cent" | "deci_cent"
    tick_size_low: float        # tick size below 0.10 (for tapered)
    tick_size_mid: float        # tick size 0.10-0.90
    tick_size_high: float       # tick size above 0.90 (for tapered)


class MarketMetadataLoader:
    """Dual-mode market metadata loader with caching."""

    def __init__(
        self,
        mode: str = "parquet",  # "parquet" or "api"
        parquet_path: str | None = None,
        api_base_url: str | None = None,
        cache_ttl_s: int = 3600,
    ):
        """Initialize loader in parquet or API mode.

        Args:
            mode: "parquet" (training) or "api" (production)
            parquet_path: Path to markets.parquet file (parquet mode)
            api_base_url: Kalshi API base URL (API mode)
            cache_ttl_s: Cache TTL in seconds (API mode only)
        """
        self._mode = mode
        self._parquet_path = parquet_path
        self._api_base_url = api_base_url
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, MarketMetadata] = {}
        self._cache_time: float = 0.0

    def load_metadata(self, tickers: list[str]) -> dict[str, MarketMetadata]:
        """Load metadata for given tickers.

        Parquet mode: read from file once, no caching needed
        API mode: fetch via REST, cache results with TTL

        Args:
            tickers: List of tickers to load metadata for

        Returns:
            Dict mapping ticker -> MarketMetadata
        """
        if self._mode == "parquet":
            return self._load_from_parquet(tickers)
        else:
            raise NotImplementedError("API mode not yet implemented")

    def _load_from_parquet(self, tickers: list[str]) -> dict[str, MarketMetadata]:
        """Load metadata from parquet file.

        Args:
            tickers: List of tickers to load

        Returns:
            Dict mapping ticker -> MarketMetadata
        """
        if self._parquet_path is None:
            raise ValueError("parquet_path required for parquet mode")

        # Load parquet file
        df = pl.read_parquet(self._parquet_path)

        # Filter to requested tickers if provided
        if tickers:
            df = df.filter(pl.col("ticker").is_in(tickers))

        # Build metadata dict
        result: dict[str, MarketMetadata] = {}
        for row in df.iter_rows(named=True):
            ticker = row["ticker"]
            structure = row["price_level_structure"]

            # Parse tick sizes from structure
            if structure == "linear_cent":
                tick_low = tick_mid = tick_high = 0.01
            elif structure == "deci_cent":
                tick_low = tick_mid = tick_high = 0.001
            elif structure == "tapered_deci_cent":
                tick_low = tick_high = 0.001  # Tails
                tick_mid = 0.01              # Middle
            else:
                # Unknown structure, use safe default
                tick_low = tick_mid = tick_high = 0.01

            result[ticker] = MarketMetadata(
                ticker=ticker,
                price_level_structure=structure,
                tick_size_low=tick_low,
                tick_size_mid=tick_mid,
                tick_size_high=tick_high,
            )

            # Cache for get_valid_tick_size calls
            self._cache[ticker] = result[ticker]

        return result

    def get_valid_tick_size(self, ticker: str, price: float) -> float:
        """Return valid tick size for this ticker at this price.

        Handles tapered markets where tick size changes by price level.
        Returns 0.001 for deci_cent, 0.01 for linear_cent, or price-dependent for tapered.

        Args:
            ticker: Market ticker
            price: Price to check (0.01-0.99)

        Returns:
            Valid tick size at this price
        """
        meta = self._cache.get(ticker)
        if meta is None:
            # Unknown ticker, assume deci_cent (conservative)
            logger.debug(f"No metadata for {ticker}, assuming deci_cent")
            return 0.001

        if meta.price_level_structure == "deci_cent":
            return 0.001
        elif meta.price_level_structure == "linear_cent":
            return 0.01
        elif meta.price_level_structure == "tapered_deci_cent":
            if price < 0.10 or price > 0.90:
                return meta.tick_size_low  # 0.001 at tails
            else:
                return meta.tick_size_mid  # 0.01 in middle
        else:
            # Unknown structure, safe default
            return 0.01

    def supports_subpenny(self, ticker: str, price: float) -> bool:
        """Check if subpenny pricing (0.001) is valid for this ticker at this price.

        Args:
            ticker: Market ticker
            price: Price to check

        Returns:
            True if 0.001 tick size is valid at this price
        """
        tick = self.get_valid_tick_size(ticker, price)
        return tick <= 0.001