"""
Fetch 3 months of historical Kalshi trades for RL model training.

Strategy: fetch ALL trades (no ticker filter) using min_ts/max_ts from both
live and historical tiers, then optionally filter client-side for a ticker
prefix. This avoids enumerating 200K+ markets individually.

Workflow:
  1. GET /historical/cutoff → learn live/historical boundary
  2. Paginate trades from historical tier (before cutoff)
  3. Paginate trades from live tier (after cutoff)
  4. Filter for ticker prefix (if specified), save trades parquet
  5. Fetch market metadata only for tickers that had trades

Output:
  output/rl_all_trades_3mo.parquet  – one row per trade
  output/rl_all_markets_3mo.parquet – one row per traded market
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import requests

from authentication_to_kalshi.auth import load_private_key, sign_pss
from config import API_KEY, KEY_PATH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://api.elections.kalshi.com"
TICKER_PREFIX = ""  # Default: fetch all markets (override with --prefix)
LOOKBACK_DAYS = 90
PAGE_LIMIT = 1000  # max per page
REQUEST_SLEEP = 0.12  # seconds between API calls

OUTPUT_DIR = Path("output")
TRADES_PARQUET = OUTPUT_DIR / "rl_all_trades_3mo.parquet"
MARKETS_PARQUET = OUTPUT_DIR / "rl_all_markets_3mo.parquet"
CHECKPOINT_PATH = OUTPUT_DIR / "trades_fetch_cursor.json"


# ---------------------------------------------------------------------------
# Authenticated request helper
# ---------------------------------------------------------------------------
_PRIVATE_KEY = None


def _get_key():
    global _PRIVATE_KEY
    if _PRIVATE_KEY is None:
        _PRIVATE_KEY = load_private_key(KEY_PATH)
    return _PRIVATE_KEY


def api_get(path: str, params: dict | None = None) -> dict:
    """Authenticated GET with retry on 429/5xx/timeouts/connection errors."""
    key = _get_key()
    for attempt in range(8):
        try:
            ts = str(int(time.time() * 1000))
            sig = sign_pss(key, ts + "GET" + path)
            headers = {
                "KALSHI-ACCESS-KEY": API_KEY,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "Content-Type": "application/json",
            }
            resp = requests.get(
                BASE_URL + path, headers=headers, params=params, timeout=45
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                print(f"  [retry] {resp.status_code} on {path}, waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = 2 ** attempt
            print(f"  [retry] {type(e).__name__} on {path}, waiting {wait}s")
            time.sleep(wait)
            continue
    # Final attempt — let exceptions propagate
    resp.raise_for_status()
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iso_to_unix(iso: str | None) -> int:
    if not iso:
        return 0
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _save_cursor_checkpoint(tier: str, cursor: str, trades_so_far: int) -> None:
    """Save pagination cursor so we can resume after interruption."""
    data = {}
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
    data[tier] = {"cursor": cursor, "trades": trades_so_far}
    CHECKPOINT_PATH.write_text(json.dumps(data))


def _load_cursor_checkpoint(tier: str) -> tuple[str | None, int]:
    """Load saved cursor for a tier. Returns (cursor, trades_so_far)."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        entry = data.get(tier, {})
        return entry.get("cursor"), entry.get("trades", 0)
    return None, 0


# ---------------------------------------------------------------------------
# Step 1: fetch cutoff
# ---------------------------------------------------------------------------
def fetch_cutoff() -> dict:
    data = api_get("/trade-api/v2/historical/cutoff")
    print(f"Cutoff: {data}")
    return data


# ---------------------------------------------------------------------------
# Step 2+3: paginate trades from a given endpoint, optionally filter by prefix
# ---------------------------------------------------------------------------
def fetch_trades_from_tier(
    path: str,
    min_ts: int,
    max_ts: int,
    tier_label: str,
    ticker_prefix: str = "",
) -> list[dict]:
    """
    Paginate all trades from `path` between min_ts and max_ts.
    Filter client-side for tickers starting with ticker_prefix (if non-empty).
    Returns list of raw trade dicts.
    """
    # Check for resume cursor
    resume_cursor, resume_count = _load_cursor_checkpoint(tier_label)
    kept: list[dict] = []
    cursor: str | None = resume_cursor
    page = 0
    total_raw = 0
    t0 = time.monotonic()

    # If resuming, load existing trades from parquet
    partial_path = OUTPUT_DIR / f"_partial_{tier_label}.parquet"
    if resume_cursor and partial_path.exists():
        df = pl.read_parquet(partial_path)
        kept = df.to_dicts()
        print(f"  Resuming {tier_label}: {len(kept)} trades from checkpoint")

    while True:
        params: dict[str, Any] = {
            "limit": PAGE_LIMIT,
            "min_ts": min_ts,
            "max_ts": max_ts,
        }
        if cursor:
            params["cursor"] = cursor

        data = api_get(path, params)
        batch = data.get("trades", [])
        if not batch:
            break

        page += 1
        total_raw += len(batch)

        # Filter for ticker prefix (if specified)
        for t in batch:
            ticker = t.get("ticker", "")
            if ticker.startswith(ticker_prefix):
                # Parse fields inline to avoid second pass
                count_str = t.get("count_fp") or t.get("count", "0")
                yes_str = t.get("yes_price_dollars") or t.get("yes_price", "0")
                no_str = t.get("no_price_dollars") or t.get("no_price", "0")
                kept.append({
                    "trade_id": t.get("trade_id"),
                    "ticker": ticker,
                    "count": float(str(count_str)),
                    "yes_price": float(str(yes_str)),
                    "no_price": float(str(no_str)),
                    "taker_side": t.get("taker_outcome_side") or t.get("taker_side"),
                    "created_time": t.get("created_time"),
                })

        # Progress every 10 pages
        if page % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = total_raw / elapsed if elapsed > 0 else 0
            filter_label = ticker_prefix if ticker_prefix else "all"
            print(f"  [{tier_label}] page {page}: {total_raw} raw, "
                  f"{len(kept)} {filter_label} | {rate:.0f} trades/s")

        cursor = data.get("cursor")
        if not cursor:
            break

        # Checkpoint every 100 pages
        if page % 100 == 0:
            _save_cursor_checkpoint(tier_label, cursor, len(kept))
            if kept:
                pl.DataFrame(kept).write_parquet(partial_path)

        time.sleep(REQUEST_SLEEP)

    elapsed = time.monotonic() - t0
    filter_label = ticker_prefix if ticker_prefix else "all"
    print(f"  [{tier_label}] done: {page} pages, {total_raw} raw trades, "
          f"{len(kept)} {filter_label} trades in {elapsed:.0f}s")

    # Clean up partial checkpoint
    if partial_path.exists():
        partial_path.unlink()

    return kept


# ---------------------------------------------------------------------------
# Step 5: fetch market metadata for traded tickers
# ---------------------------------------------------------------------------
def fetch_market_metadata(tickers: list[str]) -> list[dict]:
    """
    Fetch market details for a list of tickers using the tickers= param.
    Batch in groups of 50 (comma-separated tickers list).
    """
    all_markets: list[dict] = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        tickers_str = ",".join(batch)
        # Try live endpoint first
        data = api_get("/trade-api/v2/markets", {"tickers": tickers_str, "limit": 1000})
        markets = data.get("markets", [])
        found_set = {m["ticker"] for m in markets}
        all_markets.extend(markets)

        # For any not found in live, try historical
        missing = [t for t in batch if t not in found_set]
        if missing:
            data = api_get(
                "/trade-api/v2/historical/markets",
                {"tickers": ",".join(missing), "limit": 1000},
            )
            all_markets.extend(data.get("markets", []))

        if (i // batch_size) % 5 == 0:
            print(f"  Markets: fetched {len(all_markets)} so far "
                  f"({i + len(batch)}/{len(tickers)} tickers)")
        time.sleep(REQUEST_SLEEP)

    return all_markets


def markets_to_df(markets: list[dict]) -> pl.DataFrame:
    """Extract key fields from market dicts into a Polars DataFrame."""
    rows = []
    for m in markets:
        # Normalize volume field (could be int, string, or float)
        vol = m.get("volume") or m.get("volume_fp") or 0
        oi = m.get("open_interest") or m.get("open_interest_fp") or 0
        rows.append({
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "title": m.get("title"),
            "subtitle": m.get("subtitle") or m.get("yes_sub_title"),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time") or m.get("latest_expiration_time"),
            "status": m.get("status"),
            "result": m.get("result"),
            "volume": float(str(vol)) if vol else 0.0,
            "open_interest": float(str(oi)) if oi else 0.0,
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi trades for the past 3 months"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Ticker prefix to filter (e.g., 'KXBTC', 'INX', 'WEATHER'). Default: '' (all markets)",
    )
    args = parser.parse_args()
    ticker_prefix = args.prefix

    if not API_KEY:
        print("ERROR: PROD_API_KEY not set in .env")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    range_start = now - timedelta(days=LOOKBACK_DAYS)
    min_ts = int(range_start.timestamp())
    max_ts = int(now.timestamp())

    filter_label = ticker_prefix if ticker_prefix else "all markets"
    print(f"Fetching trades from {range_start.date()} to {now.date()} "
          f"(ts: {min_ts} – {max_ts})")
    print(f"Filter: {filter_label}")

    # 1. Get cutoff
    cutoff = fetch_cutoff()
    cutoff_raw = cutoff.get("trades_created_ts", 0)
    if isinstance(cutoff_raw, str):
        cutoff_ts = _iso_to_unix(cutoff_raw)
    else:
        cutoff_ts = int(cutoff_raw)
    print(f"Trades cutoff: {cutoff_ts} "
          f"({datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat()})")

    all_trades: list[dict] = []

    # 2. Historical trades (before cutoff)
    if min_ts < cutoff_ts:
        hist_max = min(cutoff_ts, max_ts)
        print(f"\n--- Historical tier ({range_start.date()} – "
              f"{datetime.fromtimestamp(hist_max, tz=timezone.utc).date()}) ---")
        hist_trades = fetch_trades_from_tier(
            "/trade-api/v2/historical/trades", min_ts, hist_max, "historical", ticker_prefix
        )
        all_trades.extend(hist_trades)
    else:
        print("No historical range needed (all data is live)")

    # 3. Live trades (after cutoff)
    if max_ts > cutoff_ts:
        live_min = max(cutoff_ts, min_ts)
        print(f"\n--- Live tier ({datetime.fromtimestamp(live_min, tz=timezone.utc).date()}"
              f" – {now.date()}) ---")
        live_trades = fetch_trades_from_tier(
            "/trade-api/v2/markets/trades", live_min, max_ts, "live", ticker_prefix
        )
        all_trades.extend(live_trades)
    else:
        print("No live range needed (all data is historical)")

    if not all_trades:
        print(f"\nNo trades found for filter: {filter_label}. Exiting.")
        sys.exit(1)

    # 4. Build trades DataFrame, deduplicate, save
    print(f"\nBuilding trades DataFrame ({len(all_trades)} raw rows) ...")
    df = pl.DataFrame(all_trades)
    df = df.unique(subset=["trade_id"])
    df = df.sort("created_time")
    df.write_parquet(TRADES_PARQUET)

    unique_tickers = df["ticker"].unique().to_list()
    print(f"Saved {len(df)} unique trades → {TRADES_PARQUET}")
    print(f"Date range: {df['created_time'].min()} – {df['created_time'].max()}")
    print(f"Unique tickers traded: {len(unique_tickers)}")

    # 5. Fetch market metadata for traded tickers
    print(f"\nFetching market metadata for {len(unique_tickers)} tickers ...")
    markets = fetch_market_metadata(unique_tickers)
    mkt_df = markets_to_df(markets)
    mkt_df.write_parquet(MARKETS_PARQUET)
    print(f"Saved {len(mkt_df)} markets → {MARKETS_PARQUET}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Filter: {filter_label}")
    print(f"Trades: {len(df):,}")
    print(f"Markets: {len(mkt_df):,}")
    print(f"Date range: {df['created_time'].min()} – {df['created_time'].max()}")

    # Breakdown by ticker prefix (extract first 2-6 chars before numbers/dashes)
    # Common prefixes: KXBTC, INX, WEATHER, HIGHPCP, STORMS, etc.
    ticker_prefixes = df.with_columns(
        pl.col("ticker").str.extract(r"^([A-Z]+)", 1).alias("prefix")
    )
    print("\nTrades by ticker prefix:")
    prefix_summary = ticker_prefixes.group_by("prefix").agg(
        pl.len().alias("trades"),
        pl.col("ticker").n_unique().alias("markets"),
    ).sort("trades", descending=True)
    print(prefix_summary.head(20))  # Show top 20 categories

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()
