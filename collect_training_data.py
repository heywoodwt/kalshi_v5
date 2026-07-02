"""
Collect 24h of trades + orderbook snapshots for all open Kalshi markets.

Outputs two Parquet files in output/:
  - trades_24h_YYYYMMDD_HHMMSS.parquet   (every public trade)
  - orderbooks_24h_YYYYMMDD_HHMMSS.parquet (one snapshot per market)

Usage:
    python collect_training_data.py
    python collect_training_data.py --depth 10        # deeper orderbook
    python collect_training_data.py --prefix KXBTC    # only BTC markets
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

import polars as pl

from config import API_KEY, KEY_PATH
from rl_bot.kalshi_api import KalshiRESTClient

# Rate-limit: Kalshi allows ~10 requests/sec for trading, ~30/sec for reads
REQUEST_SLEEP = 0.05  # 20 req/s, comfortably under 30/s read limit
TRADE_PAGE_LIMIT = 1000  # max per page
MARKET_PAGE_LIMIT = 200  # max per page for /markets


def build_client() -> KalshiRESTClient:
    """Build authenticated REST client from .env config."""
    return KalshiRESTClient(api_key=API_KEY, api_secret=KEY_PATH)


# ── Fetch all open markets ──────────────────────────────────────────────────
def fetch_open_markets(client: KalshiRESTClient,
                       prefix: Optional[str] = None) -> List[Dict]:
    """Page through /markets?status=open and return all market dicts."""
    markets = []
    cursor = None

    while True:
        resp = client.get_markets(limit=MARKET_PAGE_LIMIT, cursor=cursor, status="open")
        batch = resp.get("markets", [])
        if not batch:
            break

        # Optional prefix filter (client-side; API doesn't support prefix param)
        if prefix:
            batch = [m for m in batch if m.get("ticker", "").startswith(prefix)]

        markets.extend(batch)
        cursor = resp.get("cursor")

        # No more pages when cursor is empty or missing
        if not cursor:
            break

        time.sleep(REQUEST_SLEEP)

    return markets


# ── Fetch trades ─────────────────────────────────────────────────────────────
def fetch_trades_for_ticker(client: KalshiRESTClient,
                            ticker: str,
                            min_ts: int,
                            max_ts: int) -> List[Dict]:
    """Fetch all public trades for a single ticker in [min_ts, max_ts]."""
    trades = []
    cursor = None

    while True:
        resp = client.get_trades(
            ticker=ticker, min_ts=min_ts, max_ts=max_ts,
            limit=TRADE_PAGE_LIMIT, cursor=cursor,
        )
        batch = resp.get("trades", [])
        if not batch:
            break

        trades.extend(batch)
        cursor = resp.get("cursor")

        if not cursor or len(batch) < TRADE_PAGE_LIMIT:
            break

        time.sleep(REQUEST_SLEEP)

    return trades


def fetch_all_trades(client: KalshiRESTClient,
                     tickers: List[str],
                     min_ts: int,
                     max_ts: int) -> List[Dict]:
    """Fetch trades across all tickers. Prints progress."""
    all_trades = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        trades = fetch_trades_for_ticker(client, ticker, min_ts, max_ts)
        if trades:
            all_trades.extend(trades)
        # Progress every 50 tickers
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  trades: {i+1}/{total} markets scanned, {len(all_trades)} trades so far")
        time.sleep(REQUEST_SLEEP)

    return all_trades


# ── Fetch orderbooks ────────────────────────────────────────────────────────
def fetch_orderbook_snapshot(client: KalshiRESTClient,
                             ticker: str,
                             depth: int) -> Optional[Dict]:
    """Fetch and flatten one orderbook snapshot into a training-ready dict."""
    try:
        resp = client.get_orderbook(ticker, depth=depth)
    except Exception:
        return None

    book = resp.get("orderbook_fp", resp.get("orderbook", {}))
    yes_levels = book.get("yes_dollars", book.get("yes", []))
    no_levels = book.get("no_dollars", book.get("no", []))

    if not yes_levels and not no_levels:
        return None

    # Flatten into fixed-width columns: bid_0_price, bid_0_size, ...
    row = {"ticker": ticker, "fetched_at": datetime.now(timezone.utc).isoformat()}

    # YES side = bids (sorted ascending, best = last)
    for lvl in range(depth):
        idx = len(yes_levels) - 1 - lvl  # best first
        if idx >= 0:
            row[f"bid_{lvl}_price"] = float(yes_levels[idx][0])
            row[f"bid_{lvl}_size"] = float(yes_levels[idx][1])
        else:
            row[f"bid_{lvl}_price"] = 0.0
            row[f"bid_{lvl}_size"] = 0.0

    # NO side = asks (implied ask = 1 - no_bid; sorted ascending, best = last)
    for lvl in range(depth):
        idx = len(no_levels) - 1 - lvl
        if idx >= 0:
            row[f"ask_{lvl}_price"] = round(1.0 - float(no_levels[idx][0]), 4)
            row[f"ask_{lvl}_size"] = float(no_levels[idx][1])
        else:
            row[f"ask_{lvl}_price"] = 0.0
            row[f"ask_{lvl}_size"] = 0.0

    # Derived features
    if row["bid_0_price"] > 0 and row["ask_0_price"] > 0:
        row["spread"] = round(row["ask_0_price"] - row["bid_0_price"], 4)
        row["mid"] = round((row["ask_0_price"] + row["bid_0_price"]) / 2, 4)
    else:
        row["spread"] = 0.0
        row["mid"] = 0.0

    return row


def fetch_all_orderbooks(client: KalshiRESTClient,
                         tickers: List[str],
                         depth: int) -> List[Dict]:
    """Fetch orderbook snapshots for all tickers."""
    snapshots = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        snap = fetch_orderbook_snapshot(client, ticker, depth)
        if snap:
            snapshots.append(snap)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  orderbooks: {i+1}/{total} fetched, {len(snapshots)} non-empty")
        time.sleep(REQUEST_SLEEP)

    return snapshots


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Collect 24h Kalshi training data")
    parser.add_argument("--depth", type=int, default=5, help="Orderbook depth per side (default 5)")
    parser.add_argument("--prefix", type=str, default=None, help="Only collect tickers starting with PREFIX")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory (default: output)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Time window: last 24 hours (Kalshi /markets/trades uses seconds)
    now_ts = int(time.time())
    min_ts = now_ts - 24 * 60 * 60
    now_utc = datetime.now(timezone.utc)
    tag = now_utc.strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("KALSHI TRAINING DATA COLLECTOR")
    print("=" * 70)
    print(f"Window : {datetime.fromtimestamp(min_ts, tz=timezone.utc)} → {now_utc}")
    print(f"Depth  : {args.depth}")
    if args.prefix:
        print(f"Filter : {args.prefix}*")
    print()

    # 1. Build client
    client = build_client()

    # 2. Fetch open markets
    print("Fetching open markets...")
    markets = fetch_open_markets(client, prefix=args.prefix)
    tickers = [m["ticker"] for m in markets]
    print(f"  Found {len(tickers)} open markets")
    print()

    if not tickers:
        print("No markets found. Exiting.")
        return

    # 3. Fetch all trades from last 24h
    print("Fetching trades (last 24h)...")
    trades = fetch_all_trades(client, tickers, min_ts, now_ts)
    print(f"  Total trades collected: {len(trades)}")
    print()

    # 4. Fetch orderbook snapshots
    print("Fetching orderbook snapshots...")
    orderbooks = fetch_all_orderbooks(client, tickers, args.depth)
    print(f"  Total orderbook snapshots: {len(orderbooks)}")
    print()

    # 5. Save to Parquet
    if trades:
        trades_path = out_dir / f"trades_24h_{tag}.parquet"
        df_trades = pl.DataFrame(trades)
        df_trades.write_parquet(trades_path)
        print(f"Trades saved  : {trades_path}  ({len(df_trades)} rows, {df_trades.estimated_size('mb'):.1f} MB)")
    else:
        print("No trades to save.")

    if orderbooks:
        ob_path = out_dir / f"orderbooks_24h_{tag}.parquet"
        df_ob = pl.DataFrame(orderbooks)
        df_ob.write_parquet(ob_path)
        print(f"Orderbooks saved: {ob_path}  ({len(df_ob)} rows, {df_ob.estimated_size('mb'):.1f} MB)")
    else:
        print("No orderbooks to save.")

    # 6. Quick summary
    if trades:
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        df_trades = pl.DataFrame(trades)
        # Count trades per ticker
        ticker_col = "ticker" if "ticker" in df_trades.columns else "market_ticker"
        top = (
            df_trades.group_by(ticker_col)
            .agg(pl.len().alias("trade_count"))
            .sort("trade_count", descending=True)
            .head(15)
        )
        print("Top 15 most-traded markets:")
        print(top)

    print()
    print("Done.")


if __name__ == "__main__":
    main()