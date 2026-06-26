"""Generate markets metadata file from trades data.

Creates a basic markets metadata parquet with ticker and price_level_structure
columns needed by the MM metadata loader.

Usage:
    python generate_markets_metadata.py
    python generate_markets_metadata.py --trades output/rl_kalshi_trades_3mo.parquet
"""
import argparse
from pathlib import Path

import polars as pl


def infer_price_level_structure(ticker: str) -> str:
    """Infer price_level_structure from ticker pattern.

    Kalshi uses:
    - deci_cent: For most binary markets (0.001 tick at tails)
    - linear_cent: For some prediction markets (0.01 tick everywhere)
    - tapered_deci_cent: For special markets (0.001 tails, 0.01 middle)

    Default to deci_cent as it's most common for binary event markets.
    """
    # Most Kalshi markets use deci_cent or tapered_deci_cent
    # Without API data, we default to deci_cent (conservative choice)
    # This allows 0.001 tick size everywhere
    return "deci_cent"


def main():
    parser = argparse.ArgumentParser(
        description="Generate markets metadata from trades data"
    )
    parser.add_argument(
        "--trades",
        type=str,
        default="output/rl_kalshi_trades_3mo.parquet",
        help="Path to trades parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/rl_all_markets_3mo.parquet",
        help="Output path for markets metadata",
    )
    args = parser.parse_args()

    # Load trades
    print(f"Loading trades from {args.trades}")
    trades = pl.read_parquet(args.trades)
    print(f"Loaded {len(trades):,} trades")

    # Extract unique tickers
    tickers = trades.select("ticker").unique().sort("ticker")
    print(f"Found {len(tickers):,} unique tickers")

    # Add price_level_structure column
    markets = tickers.with_columns(
        pl.col("ticker")
        .map_elements(infer_price_level_structure, return_dtype=pl.Utf8)
        .alias("price_level_structure")
    )

    # Add status column (assume all are tradable since they have trades)
    markets = markets.with_columns(pl.lit("active").alias("status"))

    # Add result column (null for active markets)
    markets = markets.with_columns(pl.lit(None, dtype=pl.Utf8).alias("result"))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markets.write_parquet(output_path)

    print(f"\nCreated markets metadata: {output_path}")
    print(f"Columns: {markets.columns}")
    print(f"Rows: {len(markets):,}")
    print("\nSample:")
    print(markets.head(5))
    print("\nPrice level structure distribution:")
    print(markets.group_by("price_level_structure").count())


if __name__ == "__main__":
    main()
