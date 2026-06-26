"""Figure 1: Rolling 5-hour window of 60-second bucketed Kalshi prices for a
representative BTC strike.  HP-filter trend overlaid in solid; cycle component
(mean-reversion signal) shown in the lower panel.

Usage:
    python plot_figure1_hp_cycle.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
from statsmodels.tsa.filters.hp_filter import hpfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.tft.data_fetcher import fetch_spot_klines_with_fallback
from model.hp_dfm_rte.config import PipelineConfig

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Synthetic contract price derivation
# ---------------------------------------------------------------------------

def _spot_to_single_contract(
    spot_df: pl.DataFrame,
    strike_pct: float = 0.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Convert spot candles into a single representative binary-option series.

    strike_pct = 0.0 puts the strike at the median spot price (ATM), so the
    contract trades near 50c and has the highest gamma / most interesting
    mean-reversion dynamics.
    """
    rng = np.random.default_rng(seed)
    spot_prices = spot_df["close"].to_numpy()
    timestamps = spot_df["timestamp"].to_list()
    n = len(spot_prices)

    # Realised vol for logit scaling
    returns = np.diff(np.log(spot_prices))
    realized_vol = np.std(returns) * np.sqrt(len(returns))
    spot_mid = np.median(spot_prices)

    # ATM strike
    strike = spot_mid * (1.0 + strike_pct)
    vol_scale = spot_mid * realized_vol * 0.15
    if vol_scale < 1e-6:
        vol_scale = 100.0

    # Build price series in cents [1, 99]
    yes_prices = np.empty(n)
    for i in range(n):
        logit = (spot_prices[i] - strike) / vol_scale
        logit = np.clip(logit, -5, 5)
        prob = 1.0 / (1.0 + np.exp(-logit))
        yes_prices[i] = np.clip(prob * 98.0 + 1.0 + rng.normal(0, 0.3), 1, 99)

    return pl.DataFrame({
        "time_bucket": timestamps,
        "yes_price": yes_prices,
    }).sort("time_bucket")


# ---------------------------------------------------------------------------
# HP decomposition over the 5-hour rolling window
# ---------------------------------------------------------------------------

def decompose_hp(
    series: np.ndarray,
    hp_lambda: float = 6.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (trend, cycle) arrays from HP filter."""
    cycle, trend = hpfilter(series, lamb=hp_lambda)
    return np.asarray(trend), np.asarray(cycle)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure1(
    timestamps: list,
    prices: np.ndarray,
    trend: np.ndarray,
    cycle: np.ndarray,
    ticker_label: str,
    save_path: str = "output/figure1_hp_cycle.png",
) -> None:
    """Two-panel figure: price + trend (top), cycle (bottom)."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    # -- Top panel: Price + HP trend ------------------------------------------
    ax_top.plot(
        timestamps, prices,
        color="#7fafdf", linewidth=0.7, alpha=0.85,
        label="60 s bucketed price",
    )
    ax_top.plot(
        timestamps, trend,
        color="#d62728", linewidth=1.6,
        label="HP trend ($\\lambda=6.25$)",
    )
    ax_top.set_ylabel("Contract price (cents)", fontsize=11)
    ax_top.set_title(
        f"Figure 1 :  Rolling 5-hour window  \u2014  {ticker_label}",
        fontsize=13, fontweight="bold",
    )
    ax_top.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_top.grid(True, alpha=0.25)

    # -- Bottom panel: Cycle (mean-reversion signal) --------------------------
    # Shade positive / negative regions
    ax_bot.fill_between(
        timestamps, cycle, 0,
        where=(cycle >= 0),
        interpolate=True, color="#2ca02c", alpha=0.35,
        label="cycle > 0  (above trend)",
    )
    ax_bot.fill_between(
        timestamps, cycle, 0,
        where=(cycle < 0),
        interpolate=True, color="#d62728", alpha=0.35,
        label="cycle < 0  (below trend)",
    )
    ax_bot.plot(timestamps, cycle, color="black", linewidth=0.8, alpha=0.9)
    ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.6)

    # +/- 1 std band
    std_c = float(np.std(cycle))
    ax_bot.axhline(std_c, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_bot.axhline(-std_c, color="gray", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_bot.annotate(
        f"+1\u03c3 ({std_c:.2f}c)", xy=(timestamps[-1], std_c),
        fontsize=7, color="gray", va="bottom", ha="right",
    )
    ax_bot.annotate(
        f"\u22121\u03c3 ({-std_c:.2f}c)", xy=(timestamps[-1], -std_c),
        fontsize=7, color="gray", va="top", ha="right",
    )

    ax_bot.set_ylabel("Cycle component (cents)", fontsize=11)
    ax_bot.set_xlabel("Time (UTC)", fontsize=11)
    ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_bot.grid(True, alpha=0.25)

    # Time axis formatting
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_bot.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Figure 1: HP-filter decomposition on 60 s Kalshi buckets")
    print("=" * 60)

    # Fetch real BTC 1-min candles (5 hours = 300 candles)
    print("\nFetching BTC spot data ...")
    spot_df = await fetch_spot_klines_with_fallback(limit=300)
    print(f"  {spot_df.height} candles retrieved")

    if spot_df.height < 60:
        print("ERROR: not enough data.")
        return

    # Trim to most recent 300 rows (5 hours of 60 s buckets)
    spot_df = spot_df.tail(300)

    # Derive a single representative ATM contract
    contract_df = _spot_to_single_contract(spot_df)
    prices = contract_df["yes_price"].to_numpy()
    timestamps = contract_df["time_bucket"].to_list()

    # HP decomposition (lambda = 6.25, matching config default)
    cfg = PipelineConfig.from_env()
    trend, cycle = decompose_hp(prices, hp_lambda=cfg.hp_lambda)

    # Ticker label for the title
    ticker_label = "KXBTC (synthetic ATM strike)"

    # Plot
    plot_figure1(timestamps, prices, trend, cycle, ticker_label)

    # Quick stats
    print(f"\n  Window length : {len(prices)} buckets ({len(prices)} min)")
    print(f"  HP lambda     : {cfg.hp_lambda}")
    print(f"  Cycle std     : {np.std(cycle):.3f}c")
    print(f"  Price range   : [{prices.min():.1f}, {prices.max():.1f}]c")
    print(f"  Trend range   : [{trend.min():.1f}, {trend.max():.1f}]c")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
