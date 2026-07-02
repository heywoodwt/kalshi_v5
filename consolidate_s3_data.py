"""Consolidate raw Kalshi S3 parquets into the training format.

The kalshi-data-prod bucket (us-east-2) stores hourly parquet files partitioned
by date under trades/ and orderbooks/. This script concatenates them into two
single files the MM training pipeline consumes:

  output/rl_kalshi_trades_s3.parquet   — trades (schema already matches mm_env)
  output/s3_orderbooks.parquet         — orderbooks, column-renamed to match
                                         preprocess_mm_data()'s expectations

The trades schema (ticker, yes_price, count, taker_side, ts, created_time) is
already what preprocess_mm_data() reads, so it passes through unchanged.

The orderbook schema differs from what mm_env expects, so we rename/derive:
  implied_spread  -> spread
  yes_best_size   -> yes_size
  no_best_size    -> no_size
  yes_depth_total -> yes_depth
  no_depth_total  -> no_depth
  (derived)       -> imbalance = (yes_depth - no_depth) / (yes_depth + no_depth)
  fetched_at      -> fetched_at (unchanged; used as the window timestamp)

Usage:
    aws s3 cp s3://kalshi-data-prod/trades/      output/s3_raw/trades/      --recursive --region us-east-2
    aws s3 cp s3://kalshi-data-prod/orderbooks/  output/s3_raw/orderbooks/  --recursive --region us-east-2
    python consolidate_s3_data.py
"""
import glob

import polars as pl

RAW_TRADES = "output/s3_raw/trades/**/*.parquet"
RAW_ORDERBOOKS = "output/s3_raw/orderbooks/**/*.parquet"
OUT_TRADES = "output/rl_kalshi_trades_s3.parquet"
OUT_ORDERBOOKS = "output/s3_orderbooks.parquet"


def consolidate_trades() -> None:
    """Concatenate all trade parquets, sort by time, and write one file."""
    files = sorted(glob.glob(RAW_TRADES, recursive=True))
    if not files:
        raise FileNotFoundError(f"No trade parquets under {RAW_TRADES}")
    # Lazy scan + concat keeps memory bounded on the full set.
    df = pl.concat([pl.read_parquet(f) for f in files]).sort(["ticker", "ts"])
    df.write_parquet(OUT_TRADES)
    print(f"trades: {df.height:,} rows from {len(files)} files -> {OUT_TRADES}")


def consolidate_orderbooks() -> None:
    """Concatenate orderbook parquets and rename columns to mm_env's schema."""
    files = sorted(glob.glob(RAW_ORDERBOOKS, recursive=True))
    if not files:
        raise FileNotFoundError(f"No orderbook parquets under {RAW_ORDERBOOKS}")
    df = pl.concat([pl.read_parquet(f) for f in files])
    # Depth-based book imbalance in [-1, 1]: +1 = all bid depth, -1 = all ask depth.
    # eps guards the one-sided-book case (a depth total of 0).
    df = df.with_columns(
        (
            (pl.col("yes_depth_total") - pl.col("no_depth_total"))
            / (pl.col("yes_depth_total") + pl.col("no_depth_total") + 1e-9)
        ).alias("imbalance")
    ).rename({
        "implied_spread": "spread",
        "yes_best_size": "yes_size",
        "no_best_size": "no_size",
        "yes_depth_total": "yes_depth",
        "no_depth_total": "no_depth",
    })
    keep = ["ticker", "spread", "yes_size", "no_size", "yes_depth", "no_depth", "imbalance", "fetched_at"]
    df = df.select(keep)
    df.write_parquet(OUT_ORDERBOOKS)
    print(f"orderbooks: {df.height:,} rows from {len(files)} files -> {OUT_ORDERBOOKS}")


if __name__ == "__main__":
    consolidate_trades()
    consolidate_orderbooks()
