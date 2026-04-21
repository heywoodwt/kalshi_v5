"""Feature engineering for TFT model.

Transforms raw Kalshi + exchange data into TFT-ready features.

Target: yes_price (cents, 0-100)

Time-varying unknown reals:
    log_return, volatility_5, volatility_20, vol_ratio,
    momentum_3, momentum_10, rsi_14, price_zscore,
    spot_return, funding_rate, volume_zscore, open_interest_change

Time-varying known reals:
    time_to_expiry, hour_sin, hour_cos, minute_sin, minute_cos

Static categoricals:
    ticker

All computation in Polars; convert to pandas only at pytorch-forecasting boundary.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Regex to parse expiry from Kalshi BTC ticker
# e.g. KXBTCD-26APR1817 -> day=26, month=APR, hour=18, minute=17
_EXPIRY_RE = re.compile(
    r"KXBTC\w*-(\d{2})([A-Z]{3})(\d{2})(\d{2})"
)

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _parse_expiry_minutes(ticker: str, current_ts: float) -> float | None:
    """Parse expiry from ticker and return minutes until expiry.

    Assumes the contract expires in the current year (2026).
    """
    m = _EXPIRY_RE.search(ticker)
    if m is None:
        return None

    day = int(m.group(1))
    month = _MONTH_MAP.get(m.group(2))
    hour = int(m.group(3))
    minute = int(m.group(4))

    if month is None:
        return None

    from datetime import datetime, timezone
    try:
        expiry = datetime(2026, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None

    diff = (expiry.timestamp() - current_ts) / 60.0
    return max(0.0, diff)


def _rsi(series: pl.Series, period: int = 14) -> pl.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.clip(lower_bound=0.0)
    loss = (-delta.clip(upper_bound=0.0))

    avg_gain = gain.rolling_mean(window_size=period, min_periods=period)
    avg_loss = loss.rolling_mean(window_size=period, min_periods=period)

    # Avoid division by zero
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def engineer_features(
    df: pl.DataFrame,
    encoder_length: int = 60,
    prediction_length: int = 10,
) -> pl.DataFrame:
    """Transform raw data into TFT-ready features.

    Args:
        df: Raw data with columns:
            timestamp, ticker, yes_price, volume, spot_price,
            funding_rate, open_interest
        encoder_length: Number of historical steps for encoding.
        prediction_length: Number of future steps to predict.

    Returns:
        Long-format DataFrame with time_idx, ticker, target, and all
        feature columns. Ready for TimeSeriesDataSet construction.
    """
    if df.height == 0:
        logger.warning("Empty input DataFrame")
        return df

    # Sort by ticker then timestamp
    df = df.sort(["ticker", "timestamp"])

    # Assign time_idx per ticker (sequential integer index)
    df = df.with_columns(
        pl.col("timestamp")
        .rank("ordinal")
        .over("ticker")
        .cast(pl.Int64)
        .alias("time_idx")
    )

    # --- Time-varying unknown reals ---

    # Log return of yes_price
    df = df.with_columns(
        (pl.col("yes_price") / pl.col("yes_price").shift(1))
        .log()
        .over("ticker")
        .fill_null(0.0)
        .alias("log_return")
    )

    # Rolling volatility (5 and 20 periods)
    df = df.with_columns([
        pl.col("log_return")
        .rolling_std(window_size=5, min_periods=2)
        .over("ticker")
        .fill_null(0.0)
        .alias("volatility_5"),
        pl.col("log_return")
        .rolling_std(window_size=20, min_periods=5)
        .over("ticker")
        .fill_null(0.0)
        .alias("volatility_20"),
    ])

    # Volatility ratio (short/long)
    df = df.with_columns(
        (pl.col("volatility_5") / (pl.col("volatility_20") + 1e-10))
        .alias("vol_ratio")
    )

    # Momentum (price change over N periods)
    df = df.with_columns([
        (pl.col("yes_price") - pl.col("yes_price").shift(3))
        .over("ticker")
        .fill_null(0.0)
        .alias("momentum_3"),
        (pl.col("yes_price") - pl.col("yes_price").shift(10))
        .over("ticker")
        .fill_null(0.0)
        .alias("momentum_10"),
    ])

    # RSI (computed per ticker group)
    rsi_frames = []
    for ticker_name in df["ticker"].unique().to_list():
        mask = df.filter(pl.col("ticker") == ticker_name)
        rsi_vals = _rsi(mask["yes_price"], period=14)
        rsi_frames.append(
            mask.select("ticker", "time_idx").with_columns(
                rsi_vals.alias("rsi_14")
            )
        )
    if rsi_frames:
        rsi_df = pl.concat(rsi_frames)
        df = df.join(rsi_df, on=["ticker", "time_idx"], how="left")
    df = df.with_columns(pl.col("rsi_14").fill_null(50.0))

    # Price z-score (rolling standardization, 20-period)
    df = df.with_columns([
        (
            (pl.col("yes_price") - pl.col("yes_price").rolling_mean(window_size=20, min_periods=5).over("ticker"))
            / (pl.col("yes_price").rolling_std(window_size=20, min_periods=5).over("ticker") + 1e-10)
        )
        .fill_null(0.0)
        .alias("price_zscore"),
    ])

    # Spot return (only if spot_price column exists)
    if "spot_price" in df.columns:
        df = df.with_columns(
            (pl.col("spot_price") / pl.col("spot_price").shift(1))
            .log()
            .over("ticker")
            .fill_null(0.0)
            .alias("spot_return")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("spot_return"))

    # Funding rate default if missing
    if "funding_rate" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("funding_rate"))

    # Volume z-score (only if volume column exists)
    if "volume" in df.columns:
        df = df.with_columns(
            (
                (pl.col("volume") - pl.col("volume").rolling_mean(window_size=20, min_periods=5).over("ticker"))
                / (pl.col("volume").rolling_std(window_size=20, min_periods=5).over("ticker") + 1e-10)
            )
            .fill_null(0.0)
            .alias("volume_zscore")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("volume_zscore"))

    # Open interest change (pct change, only if column exists)
    if "open_interest" in df.columns:
        df = df.with_columns(
            (
                (pl.col("open_interest") - pl.col("open_interest").shift(1))
                / (pl.col("open_interest").shift(1).abs() + 1e-10)
            )
            .over("ticker")
            .fill_null(0.0)
            .alias("open_interest_change")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("open_interest_change"))

    # --- Time-varying known reals ---

    # Time to expiry (minutes)
    timestamps = df["timestamp"].to_list()
    tickers = df["ticker"].to_list()
    tte_values = []
    for ts, tk in zip(timestamps, tickers):
        ts_epoch = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
        tte = _parse_expiry_minutes(tk, ts_epoch)
        tte_values.append(tte if tte is not None else 60.0)  # default 60 min

    df = df.with_columns(pl.Series("time_to_expiry", tte_values))

    # Cyclical time encoding
    df = df.with_columns([
        (pl.col("timestamp").dt.hour().cast(pl.Float64) * 2.0 * math.pi / 24.0)
        .sin()
        .alias("hour_sin"),
        (pl.col("timestamp").dt.hour().cast(pl.Float64) * 2.0 * math.pi / 24.0)
        .cos()
        .alias("hour_cos"),
        (pl.col("timestamp").dt.minute().cast(pl.Float64) * 2.0 * math.pi / 60.0)
        .sin()
        .alias("minute_sin"),
        (pl.col("timestamp").dt.minute().cast(pl.Float64) * 2.0 * math.pi / 60.0)
        .cos()
        .alias("minute_cos"),
    ])

    # --- Cleanup ---

    # Replace inf/-inf with 0
    float_cols = [
        "log_return", "volatility_5", "volatility_20", "vol_ratio",
        "momentum_3", "momentum_10", "rsi_14", "price_zscore",
        "spot_return", "funding_rate", "volume_zscore", "open_interest_change",
        "time_to_expiry", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    ]
    for col in float_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(0.0)
                .otherwise(pl.col(col))
                .alias(col)
            )

    # Drop rows with null target
    df = df.filter(pl.col("yes_price").is_not_null())

    # Final column selection and ordering
    output_cols = [
        "time_idx", "ticker", "timestamp",
        # Target
        "yes_price",
        # Time-varying unknown reals
        "log_return", "volatility_5", "volatility_20", "vol_ratio",
        "momentum_3", "momentum_10", "rsi_14", "price_zscore",
        "spot_return", "funding_rate", "volume_zscore", "open_interest_change",
        # Time-varying known reals
        "time_to_expiry", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
        # Extra (kept for downstream use)
        "spot_price", "volume", "open_interest",
    ]

    # Only select columns that exist
    existing_cols = [c for c in output_cols if c in df.columns]
    df = df.select(existing_cols)

    logger.info(
        "Feature engineering complete: %d rows, %d features, %d tickers",
        df.height, len(existing_cols), df["ticker"].n_unique(),
    )
    return df


# Feature column lists (for TimeSeriesDataSet configuration)
TIME_VARYING_UNKNOWN_REALS = [
    "log_return", "volatility_5", "volatility_20", "vol_ratio",
    "momentum_3", "momentum_10", "rsi_14", "price_zscore",
    "spot_return", "funding_rate", "volume_zscore", "open_interest_change",
]

TIME_VARYING_KNOWN_REALS = [
    "time_to_expiry", "hour_sin", "hour_cos", "minute_sin", "minute_cos",
]

STATIC_CATEGORICALS = ["ticker"]

TARGET = "yes_price"