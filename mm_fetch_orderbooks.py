"""
Fetch current orderbook snapshots from the Kalshi REST API for every market
that appears in the trade data.  Produces a parquet with top-of-book prices,
spread, and depth for each active ticker.

Usage:
    python mm_fetch_orderbooks.py
    python mm_fetch_orderbooks.py --data output/rl_all_trades_3mo.parquet
    python mm_fetch_orderbooks.py --markets output/rl_all_markets_3mo.parquet
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from fetch_kalshi_trades import api_get

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 100          # max tickers per bulk request
REQUEST_SLEEP = 0.04      # ~25 req/s, under 30 read/s limit
CHECKPOINT_INTERVAL = 50  # save partial results every N batches

DEFAULT_DATA = "output/rl_kalshi_trades_3mo.parquet"
DEFAULT_OUTPUT = "output/mm_orderbooks.parquet"
CHECKPOINT_PATH = Path("output/_partial_orderbooks.parquet")


# ---------------------------------------------------------------------------
# Orderbook parsing
# ---------------------------------------------------------------------------
def _parse_levels(levels: list) -> tuple[float, float, float, int]:
    """
    Parse a side of the orderbook (yes_dollars or no_dollars).
    Each level is [price_str, size_str], sorted ascending by price.
    Best bid = highest price = last entry.
    Returns (best_price, best_size, total_depth, num_levels).
    """
    if not levels:
        return 0.0, 0.0, 0.0, 0
    # Best bid is last entry (highest price, ascending sort)
    best_price = float(levels[-1][0])
    best_size = float(levels[-1][1])
    total_depth = sum(float(lv[1]) for lv in levels)
    return best_price, best_size, total_depth, len(levels)


def _parse_orderbook(ticker: str, ob: dict, fetched_at: str) -> dict | None:
    """
    Extract top-of-book and depth from a single ticker's orderbook response.
    Returns None if both sides are empty (settled/closed market).
    """
    # Support both "orderbook_fp" wrapper and flat structure
    book = ob.get("orderbook_fp", ob)
    yes_levels = book.get("yes_dollars", book.get("yes", []))
    no_levels = book.get("no_dollars", book.get("no", []))

    # Skip tickers with completely empty orderbooks
    if not yes_levels and not no_levels:
        return None

    yes_bid, yes_size, yes_depth, yes_n = _parse_levels(yes_levels)
    no_bid, no_size, no_depth, no_n = _parse_levels(no_levels)

    # Implied spread: YES ask (= 1 - best NO bid) minus best YES bid
    # Positive spread means no arbitrage; 0 if a side is missing
    if yes_bid > 0 and no_bid > 0:
        implied_spread = (1.0 - no_bid) - yes_bid
    else:
        implied_spread = 0.0

    return {
        "ticker": ticker,
        "yes_best_bid": yes_bid,
        "yes_best_size": yes_size,
        "no_best_bid": no_bid,
        "no_best_size": no_size,
        "implied_spread": round(implied_spread, 4),
        "yes_depth_total": yes_depth,
        "no_depth_total": no_depth,
        "yes_levels": yes_n,
        "no_levels": no_n,
        "fetched_at": fetched_at,
    }


# ---------------------------------------------------------------------------
# Fetch batches
# ---------------------------------------------------------------------------
def fetch_orderbooks(tickers: list[str], depth: int = 0) -> list[dict]:
    """
    Fetch orderbook snapshots for all tickers in batches of BATCH_SIZE.
    Returns list of parsed row dicts.
    """
    results: list[dict] = []
    skipped = 0
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.monotonic()

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        batch = tickers[start : start + BATCH_SIZE]
        fetched_at = datetime.now(timezone.utc).isoformat()

        # Build params — repeated "tickers" keys for bulk endpoint
        params: dict = {"tickers": batch}
        if depth > 0:
            params["depth"] = depth

        try:
            data = api_get("/trade-api/v2/markets/orderbooks", params)
        except Exception as e:
            print(f"  [error] batch {batch_idx + 1}/{total_batches}: {e}")
            skipped += len(batch)
            time.sleep(REQUEST_SLEEP)
            continue

        # Response: {"orderbooks": {"TICKER1": {...}, "TICKER2": {...}}}
        # or possibly {"orderbooks": [{"ticker": ..., ...}, ...]}
        orderbooks = data.get("orderbooks", {})

        # Handle dict-keyed response
        if isinstance(orderbooks, dict):
            for tk, ob in orderbooks.items():
                row = _parse_orderbook(tk, ob, fetched_at)
                if row:
                    results.append(row)
                else:
                    skipped += 1
        # Handle list response
        elif isinstance(orderbooks, list):
            for ob in orderbooks:
                tk = ob.get("ticker", "")
                row = _parse_orderbook(tk, ob, fetched_at)
                if row:
                    results.append(row)
                else:
                    skipped += 1

        # Progress
        elapsed = time.monotonic() - t0
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  batch {batch_idx + 1}/{total_batches}: "
                  f"{len(results)} fetched, {skipped} skipped "
                  f"({elapsed:.1f}s)")

        # Checkpoint partial results
        if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0 and results:
            pl.DataFrame(results).write_parquet(CHECKPOINT_PATH)
            print(f"  [checkpoint] saved {len(results)} rows")

        time.sleep(REQUEST_SLEEP)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch orderbook snapshots for traded Kalshi markets"
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA,
        help="Path to trades parquet (default: output/rl_kalshi_trades_3mo.parquet)",
    )
    parser.add_argument(
        "--markets", type=str, default=None,
        help="Path to markets metadata parquet (optional, for pre-filtering settled)",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output parquet path (default: output/mm_orderbooks.parquet)",
    )
    parser.add_argument(
        "--depth", type=int, default=0,
        help="Orderbook depth per ticker (0 = all levels)",
    )
    args = parser.parse_args()

    # 1. Load trade data and extract unique tickers
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: trade data not found: {data_path}")
        sys.exit(1)

    trades = pl.read_parquet(data_path)
    all_tickers = trades["ticker"].unique().sort().to_list()
    print(f"Loaded {len(trades):,} trades with {len(all_tickers):,} unique tickers")

    # 2. Pre-filter settled markets if metadata provided
    if args.markets:
        markets_path = Path(args.markets)
        if markets_path.exists():
            mkt = pl.read_parquet(markets_path)
            # Keep only tickers that are NOT settled
            settled = mkt.filter(
                (pl.col("status") == "settled") | pl.col("result").is_not_null()
            )["ticker"].to_list()
            settled_set = set(settled)
            before = len(all_tickers)
            all_tickers = [t for t in all_tickers if t not in settled_set]
            print(f"Filtered out {before - len(all_tickers)} settled markets "
                  f"→ {len(all_tickers)} remaining")
        else:
            print(f"Warning: markets file not found: {markets_path}, skipping filter")

    if not all_tickers:
        print("No tickers to fetch. Exiting.")
        sys.exit(0)

    # 3. Fetch orderbooks
    print(f"\nFetching orderbooks for {len(all_tickers)} tickers "
          f"(batches of {BATCH_SIZE}) ...")
    results = fetch_orderbooks(all_tickers, args.depth)

    if not results:
        print("\nNo orderbook data returned (all markets may be settled). Exiting.")
        sys.exit(0)

    # 4. Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(results)
    df = df.sort("ticker")
    df.write_parquet(output_path)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    # 5. Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Tickers attempted: {len(all_tickers):,}")
    print(f"Orderbooks fetched: {len(df):,}")
    print(f"Skipped (empty/settled): {len(all_tickers) - len(df):,}")
    print(f"Output: {output_path}")

    # Spread stats for non-zero spreads
    valid = df.filter(pl.col("implied_spread") > 0)
    if len(valid) > 0:
        spread_stats = valid["implied_spread"].describe()
        print(f"\nImplied spread (active markets, n={len(valid)}):")
        print(f"  mean:   {valid['implied_spread'].mean():.4f}")
        print(f"  median: {valid['implied_spread'].median():.4f}")
        print(f"  min:    {valid['implied_spread'].min():.4f}")
        print(f"  max:    {valid['implied_spread'].max():.4f}")

    # Depth stats
    print(f"\nYES depth total:  mean={df['yes_depth_total'].mean():.1f}, "
          f"median={df['yes_depth_total'].median():.1f}")
    print(f"NO depth total:   mean={df['no_depth_total'].mean():.1f}, "
          f"median={df['no_depth_total'].median():.1f}")


if __name__ == "__main__":
    main()
