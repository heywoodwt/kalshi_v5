"""Fetch crypto market data for TFT training.

Builds a consolidated Polars DataFrame with columns:
    timestamp, ticker, yes_price, volume, spot_price, funding_rate, open_interest

Exchange data fetchers with fallback chains:
    Spot:     Coinbase -> Kraken -> Bybit -> Binance
    Funding:  Bybit -> Binance
    OI:       Bybit -> Binance

Kalshi trade data must be provided via parquet file or WebSocket accumulation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import httpx
import polars as pl

logger = logging.getLogger(__name__)

_BINANCE_BASE = "https://fapi.binance.com"
_SPOT_BASE = "https://api.binance.com"
_BYBIT_BASE = "https://api.bybit.com"
_COINBASE_BASE = "https://api.exchange.coinbase.com"
_KRAKEN_BASE = "https://api.kraken.com"
_TIMEOUT = 10.0


def _empty_funding_df() -> pl.DataFrame:
    return pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "funding_rate": pl.Float64})


def _empty_oi_df() -> pl.DataFrame:
    return pl.DataFrame(schema={"timestamp": pl.Datetime("us", "UTC"), "open_interest": pl.Float64})


# ---------------------------------------------------------------------------
# Spot price fetchers (with fallback chain)
# ---------------------------------------------------------------------------

async def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 1500,
) -> pl.DataFrame:
    """Fetch spot klines from Binance public API."""
    params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_SPOT_BASE}/api/v3/klines", params=params)
        resp.raise_for_status()
        data = resp.json()

    rows = []
    for k in data:
        rows.append({
            "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return pl.DataFrame(rows)


async def fetch_coinbase_klines(
    product_id: str = "BTC-USD",
    granularity_s: int = 60,
    limit: int = 300,
) -> pl.DataFrame:
    """Fetch spot candles from Coinbase Exchange API."""
    params = {"granularity": granularity_s}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_COINBASE_BASE}/products/{product_id}/candles", params=params)
        resp.raise_for_status()
        data = resp.json()

    rows = []
    for row in data[:limit]:
        rows.append({
            "timestamp": datetime.fromtimestamp(float(row[0]), tz=timezone.utc),
            "open": float(row[3]),
            "high": float(row[2]),
            "low": float(row[1]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return pl.DataFrame(rows).sort("timestamp") if rows else pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
    )


async def fetch_kraken_klines(pair: str = "XBTUSD", interval: int = 1) -> pl.DataFrame:
    """Fetch spot OHLC data from Kraken public API."""
    params = {"pair": pair, "interval": interval}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_KRAKEN_BASE}/0/public/OHLC", params=params)
        resp.raise_for_status()
        payload = resp.json()

    result = payload.get("result", {})
    key = next((k for k in result.keys() if k != "last"), None)
    if not key:
        return pl.DataFrame(
            schema={"timestamp": pl.Datetime("us", "UTC"), "open": pl.Float64,
                    "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
        )

    rows = []
    for row in result.get(key, []):
        rows.append({
            "timestamp": datetime.fromtimestamp(float(row[0]), tz=timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]),
        })
    return pl.DataFrame(rows).sort("timestamp") if rows else pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
    )


async def fetch_bybit_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1",
    limit: int = 1000,
) -> pl.DataFrame:
    """Fetch kline data from Bybit public API."""
    params = {"category": "linear", "symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BYBIT_BASE}/v5/market/kline", params=params)
        resp.raise_for_status()
        payload = resp.json()

    rows = []
    for row in payload.get("result", {}).get("list", []):
        rows.append({
            "timestamp": datetime.fromtimestamp(float(row[0]) / 1000.0, tz=timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return pl.DataFrame(rows).sort("timestamp") if rows else pl.DataFrame(
        schema={"timestamp": pl.Datetime("us", "UTC"), "open": pl.Float64,
                "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64}
    )


# ---------------------------------------------------------------------------
# Funding rate & open interest fetchers
# ---------------------------------------------------------------------------

async def fetch_binance_funding_rate(symbol: str = "BTCUSDT", limit: int = 500) -> pl.DataFrame:
    """Fetch funding rate history from Binance Futures API."""
    params: dict[str, str | int] = {"symbol": symbol, "limit": limit}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BINANCE_BASE}/fapi/v1/fundingRate", params=params)
        resp.raise_for_status()
        data = resp.json()

    rows = [{"timestamp": datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc),
             "funding_rate": float(r["fundingRate"])} for r in data]
    return pl.DataFrame(rows) if rows else _empty_funding_df()


async def fetch_bybit_funding_rate(symbol: str = "BTCUSDT", limit: int = 200) -> pl.DataFrame:
    """Fetch funding rate history from Bybit futures API."""
    params = {"category": "linear", "symbol": symbol, "limit": limit}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BYBIT_BASE}/v5/market/funding/history", params=params)
        resp.raise_for_status()
        payload = resp.json()

    rows = [{"timestamp": datetime.fromtimestamp(float(item["fundingRateTimestamp"]) / 1000.0, tz=timezone.utc),
             "funding_rate": float(item["fundingRate"])}
            for item in payload.get("result", {}).get("list", [])]
    return pl.DataFrame(rows).sort("timestamp") if rows else _empty_funding_df()


async def fetch_binance_open_interest(symbol: str = "BTCUSDT", period: str = "5m", limit: int = 500) -> pl.DataFrame:
    """Fetch open interest history from Binance Futures API."""
    params: dict[str, str | int] = {"symbol": symbol, "period": period, "limit": limit}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BINANCE_BASE}/futures/data/openInterestHist", params=params)
        resp.raise_for_status()
        data = resp.json()

    rows = [{"timestamp": datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc),
             "open_interest": float(r["sumOpenInterest"])} for r in data]
    return pl.DataFrame(rows) if rows else _empty_oi_df()


async def fetch_bybit_open_interest(symbol: str = "BTCUSDT", interval: str = "5min", limit: int = 200) -> pl.DataFrame:
    """Fetch open interest history from Bybit futures API."""
    params = {"category": "linear", "symbol": symbol, "intervalTime": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BYBIT_BASE}/v5/market/open-interest", params=params)
        resp.raise_for_status()
        payload = resp.json()

    rows = [{"timestamp": datetime.fromtimestamp(float(item["timestamp"]) / 1000.0, tz=timezone.utc),
             "open_interest": float(item["openInterest"])}
            for item in payload.get("result", {}).get("list", [])]
    return pl.DataFrame(rows).sort("timestamp") if rows else _empty_oi_df()


# ---------------------------------------------------------------------------
# Fallback chains
# ---------------------------------------------------------------------------

async def fetch_spot_klines_with_fallback(limit: int = 1500) -> pl.DataFrame:
    """Fetch BTC spot candles with multi-exchange fallback."""
    providers = [
        ("coinbase", lambda: fetch_coinbase_klines(limit=min(limit, 300))),
        ("kraken", fetch_kraken_klines),
        ("bybit", lambda: fetch_bybit_klines(limit=min(limit, 1000))),
        ("binance", lambda: fetch_binance_klines(limit=limit)),
    ]
    for name, fn in providers:
        try:
            df = await fn()
            if df.height > 0:
                logger.info("Using %s spot data (%d rows)", name, df.height)
                return df
        except Exception as exc:
            logger.warning("Spot provider %s failed: %s", name, type(exc).__name__)
    raise RuntimeError("No spot provider available")


async def fetch_funding_rate_with_fallback() -> pl.DataFrame:
    """Fetch funding rate with fallback; return empty on total failure."""
    for name, fn in [("bybit", fetch_bybit_funding_rate), ("binance", fetch_binance_funding_rate)]:
        try:
            df = await fn()
            if df.height > 0:
                logger.info("Using %s funding data (%d rows)", name, df.height)
                return df
        except Exception as exc:
            logger.warning("Funding provider %s failed: %s", name, type(exc).__name__)
    logger.warning("No funding source available — using zeros")
    return _empty_funding_df()


async def fetch_open_interest_with_fallback() -> pl.DataFrame:
    """Fetch open interest with fallback; return empty on total failure."""
    for name, fn in [("bybit", fetch_bybit_open_interest), ("binance", fetch_binance_open_interest)]:
        try:
            df = await fn()
            if df.height > 0:
                logger.info("Using %s open interest data (%d rows)", name, df.height)
                return df
        except Exception as exc:
            logger.warning("Open interest provider %s failed: %s", name, type(exc).__name__)
    logger.warning("No open interest source available — using zeros")
    return _empty_oi_df()


# ---------------------------------------------------------------------------
# Consolidated data builder
# ---------------------------------------------------------------------------

async def build_training_data(
    kalshi_df: pl.DataFrame,
    binance_kline_limit: int = 1500,
) -> pl.DataFrame:
    """Build consolidated training DataFrame from Kalshi trades + exchange data.

    Args:
        kalshi_df: Kalshi trade data with columns: timestamp, ticker, yes_price, volume
                   (from WebSocket accumulation or saved parquet file)
        binance_kline_limit: Number of klines to fetch for spot data.

    Returns:
        Long-format Polars DataFrame with columns:
            timestamp, ticker, yes_price, volume, spot_price, funding_rate, open_interest
    """
    if kalshi_df.height == 0:
        logger.warning("No Kalshi trades provided — returning empty DataFrame")
        return pl.DataFrame(schema={
            "timestamp": pl.Datetime("us", "UTC"), "ticker": pl.Utf8,
            "yes_price": pl.Float64, "volume": pl.Float64,
            "spot_price": pl.Float64, "funding_rate": pl.Float64, "open_interest": pl.Float64,
        })

    # Fetch exchange data in parallel
    klines_df, funding_df, oi_df = await asyncio.gather(
        fetch_spot_klines_with_fallback(limit=binance_kline_limit),
        fetch_funding_rate_with_fallback(),
        fetch_open_interest_with_fallback(),
    )

    # Bucket Kalshi trades to 1-min intervals per ticker (VWAP)
    kalshi_bucketed = kalshi_df
    if "trade_count" in kalshi_df.columns:
        kalshi_bucketed = (
            kalshi_df
            .with_columns(pl.col("timestamp").dt.truncate("1m").alias("timestamp_bucket"))
            .group_by(["timestamp_bucket", "ticker"])
            .agg([
                (pl.col("yes_price") * pl.col("trade_count")).sum()
                .truediv(pl.col("trade_count").sum())
                .alias("yes_price"),
                pl.col("trade_count").sum().alias("volume"),
            ])
            .rename({"timestamp_bucket": "timestamp"})
            .sort(["ticker", "timestamp"])
        )

    # Join spot price via asof join
    klines_join = klines_df.select([
        pl.col("timestamp").dt.truncate("1m").alias("timestamp"),
        pl.col("close").alias("spot_price"),
    ])
    result = kalshi_bucketed.sort("timestamp").join_asof(
        klines_join.sort("timestamp"), on="timestamp", strategy="backward",
    )

    # Join funding rate
    if funding_df.height > 0:
        funding_join = funding_df.select([
            pl.col("timestamp").dt.truncate("1m"), pl.col("funding_rate"),
        ]).sort("timestamp")
        result = result.sort("timestamp").join_asof(funding_join, on="timestamp", strategy="backward")
    else:
        result = result.with_columns(pl.lit(0.0).alias("funding_rate"))

    # Join open interest
    if oi_df.height > 0:
        oi_join = oi_df.select([
            pl.col("timestamp").dt.truncate("1m"), pl.col("open_interest"),
        ]).sort("timestamp")
        result = result.sort("timestamp").join_asof(oi_join, on="timestamp", strategy="backward")
    else:
        result = result.with_columns(pl.lit(0.0).alias("open_interest"))

    # Ensure column order
    result = result.select([
        "timestamp", "ticker", "yes_price", "volume",
        "spot_price", "funding_rate", "open_interest",
    ])

    # Fill remaining nulls
    result = result.with_columns([
        pl.col("spot_price").forward_fill().backward_fill(),
        pl.col("funding_rate").forward_fill().fill_null(0.0),
        pl.col("open_interest").forward_fill().fill_null(0.0),
        pl.col("volume").fill_null(0.0),
    ])

    logger.info(
        "Built training data: %d rows, %d tickers, range %s to %s",
        result.height, result["ticker"].n_unique(),
        result["timestamp"].min(), result["timestamp"].max(),
    )
    return result


# ---------------------------------------------------------------------------
# Synthetic data for testing
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_tickers: int = 5,
    n_steps: int = 500,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic training data for testing the TFT pipeline.

    Creates realistic-looking Kalshi BTC contract prices with correlated
    spot prices and external features.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    base_time = datetime(2026, 4, 15, 0, 0, tzinfo=timezone.utc)

    for t_idx in range(n_tickers):
        ticker = f"KXBTCD-26APR18{17 + t_idx:02d}"
        spot_base = 85000.0 + rng.normal(0, 500)
        price_base = 50.0 + rng.normal(0, 10)

        for step in range(n_steps):
            ts = base_time.replace(minute=step % 60, hour=(step // 60) % 24)

            # Random walk for spot price
            spot_base += rng.normal(0, 50)
            spot = max(spot_base, 50000)

            # Price follows spot + noise
            price_base += rng.normal(0, 1.5)
            yes_price = max(1.0, min(99.0, price_base))

            rows.append({
                "timestamp": ts,
                "ticker": ticker,
                "yes_price": yes_price,
                "volume": float(max(0, rng.poisson(5))),
                "spot_price": spot,
                "funding_rate": rng.normal(0.0001, 0.0005),
                "open_interest": max(0.0, 100000 + rng.normal(0, 5000)),
            })

    df = pl.DataFrame(rows).sort(["ticker", "timestamp"])
    logger.info("Generated synthetic data: %d rows, %d tickers", df.height, n_tickers)
    return df