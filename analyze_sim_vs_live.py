"""Sim-vs-live calibration: measure the adverse-selection bias in the MM simulator.

The training/eval simulator (rl_bot/mm_env.py) credits a fill instantly at the
window mid price. That is, when the bot buys at the bid it books +half_spread of
paper profit right away, and it never lets the price move against the new position.
Real market making does not work this way: the counterparty who filled you is
disproportionately right, so the mid drifts against your inventory after the fill.
That drift is the adverse-selection cost the simulator ignores, and it is why sim
PnL is a badly upward-biased estimator of live PnL.

This script measures that bias directly from real fills using a markout analysis.

For every live fill we compute, in YES-equivalent terms:

    entry_edge   = signed * (mid_before_fill - fill_price)     # what the sim credits
    markout(dt)  = signed * (mid_at_fill_ts+dt - fill_price)   # realizable edge later
    adverse(dt)  = entry_edge - markout(dt)
                 = signed * (mid_before_fill - mid_at_fill_ts+dt)

`signed` is +1 if the fill made us longer YES, -1 if shorter. `entry_edge` is the
spread the sim pays us. `markout(dt)` is what that edge is actually worth once the
market has moved `dt` seconds later. `adverse(dt)` is the shortfall — the per-contract
bias. Aggregated per category, adverse(dt) is exactly the "sim minus live" gap.

Mid prices come from the trades parquet (last trade price as a mid proxy — the same
price series the simulator itself marks against). The mid *before* the fill is taken
strictly before the fill timestamp so the fill's own print cannot leak into it.

Usage:
    # Fetch recent fills from Kalshi API (needs KALSHI_* env vars), then analyse:
    python analyze_sim_vs_live.py --hours 48

    # Re-use cached fills, no API call:
    python analyze_sim_vs_live.py --fills-parquet output/live_fills.parquet

Output: a per-(category, horizon) CSV and a printed summary table.
"""

import argparse
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl

from rl_bot.reward import compute_maker_fee

# Kalshi maker fee rate used across the codebase (see rl_bot/live_trader_v2.py).
MAKER_FEE_RATE = 0.0175


# --- Fill loading ---------------------------------------------------------------


def _parse_iso_to_ms(iso: str) -> int:
    """Convert an ISO-8601 timestamp string to epoch milliseconds."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def normalize_fills(raw_fills: list[dict]) -> pl.DataFrame:
    """Convert raw Kalshi fill dicts into a typed, YES-equivalent Polars frame.

    Kalshi fills carry an `action` (buy/sell) and a `side` (yes/no). We fold both
    into a single YES-equivalent view so long/short and price are directly
    comparable across all four combinations:

        buy  yes -> long YES  (+1), price = yes_price
        sell no  -> long YES  (+1), price = 1 - no_price
        sell yes -> short YES (-1), price = yes_price
        buy  no  -> short YES (-1), price = 1 - no_price
    """
    ts_ms: list[int] = []
    tickers: list[str] = []
    categories: list[str] = []
    signed: list[int] = []
    price_yes: list[float] = []
    counts: list[float] = []

    for f in raw_fills:
        ticker = f.get("market_ticker", "")
        if not ticker:
            continue
        action = f.get("action", "")
        side = f.get("side", "")
        if action not in ("buy", "sell") or side not in ("yes", "no"):
            continue

        # YES-equivalent price: a NO price of p means YES trades at 1 - p.
        if side == "yes":
            p = float(f.get("yes_price_dollars", 0.0))
        else:
            p = 1.0 - float(f.get("no_price_dollars", 0.0))

        # Direction in YES terms: +1 when the fill lengthens our YES position.
        # (buy==yes) or (sell==no) both mean long YES.
        s = 1 if (action == "buy") == (side == "yes") else -1

        # Timestamp: prefer numeric `ts`; fall back to parsing created_time.
        t = f.get("ts")
        if t is None:
            created = f.get("created_time")
            if not created:
                continue
            t = _parse_iso_to_ms(created)
        else:
            t = int(t)
            # The API returns epoch seconds; normalize to milliseconds.
            # (Anything below 1e12 cannot be a millisecond timestamp after 2001.)
            if t < 10**12:
                t *= 1000

        ts_ms.append(int(t))
        tickers.append(ticker)
        categories.append(ticker.split("-")[0])  # series prefix = category
        signed.append(s)
        price_yes.append(p)
        counts.append(float(f.get("count_fp", f.get("count", 1))))

    return pl.DataFrame({
        "ts_ms": ts_ms,
        "ticker": tickers,
        "category": categories,
        "signed": signed,
        "price_yes": price_yes,
        "count": counts,
    })


def fetch_fills(hours: float, cache_path: str) -> pl.DataFrame:
    """Fetch fills from the Kalshi API over the last `hours` and cache them.

    The API has no cursor in this client, so we page by splitting the lookback
    into 1-hour windows and requesting up to 1000 fills per window.
    """
    import os

    from dotenv import load_dotenv

    from rl_bot.kalshi_api import KalshiRESTClient

    load_dotenv()
    client = KalshiRESTClient(
        api_key=os.getenv("KALSHI_API_KEY"),
        api_secret=os.getenv("KALSHI_API_SECRET"),
    )
    email, password = os.getenv("KALSHI_EMAIL"), os.getenv("KALSHI_PASSWORD")
    if email and password:
        client.login(email, password)

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(hours * 3600 * 1000)
    window_ms = 3600 * 1000  # 1-hour windows

    raw: list[dict] = []
    w = start_ms
    while w < now_ms:
        resp = client.get_fills(min_ts=w // 1000, max_ts=min(w + window_ms, now_ms) // 1000, limit=1000)
        raw.extend(resp.get("fills", []))
        w += window_ms

    df = normalize_fills(raw)
    if len(df) > 0:
        df.write_parquet(cache_path)
        print(f"Fetched {len(df)} fills -> cached to {cache_path}")
    else:
        print("No fills returned by API.")
    return df


# --- Mid-price index ------------------------------------------------------------


def build_mid_index(trades_df: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build a per-ticker step-function mid index from trade prints.

    Returns a dict mapping ticker -> (ts_ms_sorted, yes_price). The last trade at
    or before a query time is used as the mid — the same last/VWAP price basis the
    simulator marks against, so the comparison is fair.
    """
    trades_df = trades_df.select(["ticker", "ts", "yes_price"]).sort(["ticker", "ts"])
    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # partition_by keeps each ticker's rows contiguous and already time-sorted.
    for part in trades_df.partition_by("ticker", maintain_order=True):
        ticker = part["ticker"][0]
        index[ticker] = (
            part["ts"].to_numpy().astype(np.int64),
            part["yes_price"].to_numpy().astype(np.float64),
        )
    return index


def mid_at(index_entry: tuple[np.ndarray, np.ndarray], t_ms: int, strict_before: bool) -> float | None:
    """Look up the mid (last trade price) at or before `t_ms` via binary search.

    strict_before=True returns the last trade strictly before t_ms (used for the
    pre-fill mid so the fill's own print cannot leak in). Returns None if there is
    no trade before the query time.
    """
    ts_arr, px_arr = index_entry
    # searchsorted 'left'  -> first index with ts >= t   (strictly-before = idx-1)
    # searchsorted 'right' -> first index with ts >  t   (at-or-before   = idx-1)
    side = "left" if strict_before else "right"
    idx = int(np.searchsorted(ts_arr, t_ms, side=side)) - 1
    if idx < 0:
        return None
    return float(px_arr[idx])


# --- Calibration ----------------------------------------------------------------


def calibrate(fills: pl.DataFrame, mid_index: dict, horizons_s: list[int]) -> pl.DataFrame:
    """Compute per-fill entry edge, markout, and adverse selection for each horizon.

    Returns a long-format frame with one row per (fill, horizon).
    """
    out_rows: dict[str, list] = {
        "fill_id": [], "category": [], "ticker": [], "ts_ms": [], "count": [],
        "horizon_s": [], "entry_edge": [], "markout": [], "adverse": [],
        "maker_fee_per_contract": [],
    }

    # Pull columns to numpy/lists once (fast, no per-row Polars overhead).
    ts_col = fills["ts_ms"].to_list()
    tk_col = fills["ticker"].to_list()
    cat_col = fills["category"].to_list()
    sg_col = fills["signed"].to_list()
    pz_col = fills["price_yes"].to_list()
    ct_col = fills["count"].to_list()

    for i in range(len(fills)):
        entry_idx = mid_index.get(tk_col[i])
        if entry_idx is None:
            continue  # no trade data for this ticker — cannot mark

        mid_before = mid_at(entry_idx, ts_col[i], strict_before=True)
        if mid_before is None:
            continue  # fill precedes all available trade prints

        s = sg_col[i]
        price = pz_col[i]
        entry_edge = s * (mid_before - price)
        fee_pc = compute_maker_fee(1, price, MAKER_FEE_RATE)  # per-contract maker fee

        for h in horizons_s:
            mid_future = mid_at(entry_idx, ts_col[i] + h * 1000, strict_before=False)
            if mid_future is None:
                continue
            markout = s * (mid_future - price)
            out_rows["fill_id"].append(i)
            out_rows["category"].append(cat_col[i])
            out_rows["ticker"].append(tk_col[i])
            out_rows["ts_ms"].append(ts_col[i])
            out_rows["count"].append(ct_col[i])
            out_rows["horizon_s"].append(h)
            out_rows["entry_edge"].append(entry_edge)
            out_rows["markout"].append(markout)
            out_rows["adverse"].append(entry_edge - markout)
            out_rows["maker_fee_per_contract"].append(fee_pc)

    return pl.DataFrame(out_rows)


def aggregate(per_fill: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-fill markouts into per-(category, horizon) calibration stats.

    All edges are per-contract dollar amounts; totals weight by fill count. The key
    columns:
      sim_edge_total  = spread the simulator credits (entry_edge * count), summed
      live_edge_total = realizable edge after the market moves (markout * count)
      bias_total      = sim_edge_total - live_edge_total = adverse selection paid
      net_live_total  = live_edge_total minus entry maker fees (closer to real PnL)
    """
    return (
        per_fill.group_by(["category", "horizon_s"])
        .agg([
            pl.len().alias("n_fills"),
            pl.col("count").sum().alias("contracts"),
            (pl.col("entry_edge") * pl.col("count")).sum().alias("sim_edge_total"),
            (pl.col("markout") * pl.col("count")).sum().alias("live_edge_total"),
            (pl.col("adverse") * pl.col("count")).sum().alias("bias_total"),
            (pl.col("maker_fee_per_contract") * pl.col("count")).sum().alias("maker_fee_total"),
            pl.col("entry_edge").mean().alias("sim_edge_per_contract"),
            pl.col("markout").mean().alias("live_edge_per_contract"),
            pl.col("adverse").mean().alias("adverse_per_contract"),
        ])
        .with_columns(
            (pl.col("live_edge_total") - pl.col("maker_fee_total")).alias("net_live_total")
        )
        .sort(["category", "horizon_s"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim-vs-live MM calibration (markout analysis)")
    parser.add_argument("--fills-parquet", default=None,
                        help="Load cached fills instead of hitting the Kalshi API")
    parser.add_argument("--trades", default="output/rl_kalshi_trades_june.parquet",
                        help="Trades parquet used to build the mid-price series")
    parser.add_argument("--hours", type=float, default=48.0,
                        help="Lookback window for API fill fetch (ignored with --fills-parquet)")
    parser.add_argument("--horizons", default="30,60,300",
                        help="Comma-separated markout horizons in seconds")
    parser.add_argument("--out", default="output/sim_vs_live_calibration.csv",
                        help="Output CSV path for per-(category,horizon) stats")
    parser.add_argument("--fills-cache", default="output/live_fills.parquet",
                        help="Where to cache API-fetched fills")
    args = parser.parse_args()

    horizons_s = [int(x) for x in args.horizons.split(",") if x.strip()]

    # 1. Load fills (cached parquet or live API).
    if args.fills_parquet:
        fills = pl.read_parquet(args.fills_parquet)
        print(f"Loaded {len(fills)} fills from {args.fills_parquet}")
    else:
        fills = fetch_fills(args.hours, args.fills_cache)
    if len(fills) == 0:
        print("No fills to analyse — exiting.")
        return

    # 2. Build the mid-price index from trade prints.
    print(f"Loading trades from {args.trades} ...")
    trades = pl.read_parquet(args.trades, columns=["ticker", "ts", "yes_price"])
    mid_index = build_mid_index(trades)
    print(f"Mid index built for {len(mid_index)} tickers.")

    # 3. Per-fill markout and 4. aggregation.
    per_fill = calibrate(fills, mid_index, horizons_s)
    matched = per_fill["fill_id"].n_unique() if len(per_fill) else 0
    print(f"Matched {matched}/{len(fills)} fills to trade data "
          f"({len(fills) - matched} unmatched — no overlapping trade prints).")
    if len(per_fill) == 0:
        print("No fills could be marked against trade data — check ticker/time overlap.")
        return

    agg = aggregate(per_fill)
    agg.write_csv(args.out)

    # 5. Print a readable summary. All figures are dollars (contracts priced 0..1).
    print("\n" + "=" * 92)
    print("SIM-VS-LIVE CALIBRATION  (positive bias = simulator overstates edge)")
    print("=" * 92)
    print(f"{'category':<22}{'dt_s':>5}{'fills':>7}{'sim/ct':>10}{'live/ct':>10}"
          f"{'adv/ct':>10}{'bias_$':>10}{'net_live$':>11}")
    print("-" * 92)
    for r in agg.iter_rows(named=True):
        print(f"{r['category']:<22}{r['horizon_s']:>5}{r['n_fills']:>7}"
              f"{r['sim_edge_per_contract']:>10.4f}{r['live_edge_per_contract']:>10.4f}"
              f"{r['adverse_per_contract']:>10.4f}{r['bias_total']:>10.2f}{r['net_live_total']:>11.2f}")

    # Portfolio-level totals per horizon.
    print("-" * 92)
    totals = (
        agg.group_by("horizon_s")
        .agg([
            pl.col("sim_edge_total").sum(),
            pl.col("live_edge_total").sum(),
            pl.col("bias_total").sum(),
            pl.col("net_live_total").sum(),
        ])
        .sort("horizon_s")
    )
    for r in totals.iter_rows(named=True):
        print(f"{'ALL':<22}{r['horizon_s']:>5}{'':>7}"
              f"{'':>10}{'':>10}{'':>10}{r['bias_total']:>10.2f}{r['net_live_total']:>11.2f}"
              f"   (sim_edge={r['sim_edge_total']:.2f})")
    print("=" * 92)
    print(f"\nPer-(category,horizon) detail written to {args.out}")
    print("Read: sim/ct is what mm_env.py credits per contract; live/ct is the edge that\n"
          "survives after dt seconds; adv/ct is the adverse-selection gap the sim ignores.")


if __name__ == "__main__":
    main()
