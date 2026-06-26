"""Figure 2: One-step HP-DFM cycle forecast vs. realised cycle with +/-1 sigma
band and entry/exit markers overlaid.

Usage:
    python plot_figure2_forecast_vs_realised.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.tft.data_fetcher import fetch_spot_klines_with_fallback
from model.hp_dfm_rte.config import PipelineConfig
from model.hp_dfm_rte.model_engine import make_engine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic contract derivation (reused from plot_hp_dfm_predictions.py)
# ---------------------------------------------------------------------------

def _spot_to_contract_prices(
    spot_df: pl.DataFrame,
    n_contracts: int = 3,
    seed: int = 42,
) -> pl.DataFrame:
    """Derive realistic Kalshi binary-option series from real BTC spot candles."""
    rng = np.random.default_rng(seed)
    spot_prices = spot_df["close"].to_numpy()
    timestamps = spot_df["timestamp"].to_list()
    n = len(spot_prices)

    returns = np.diff(np.log(spot_prices))
    realized_vol = np.std(returns) * np.sqrt(len(returns))
    spot_mid = np.median(spot_prices)

    rows: list[dict] = []
    for c_idx in range(n_contracts):
        strike_offset = (c_idx - n_contracts // 2) * spot_mid * realized_vol * 0.3
        strike = spot_mid + strike_offset
        ticker = f"KXBTCD-26APR{17 + c_idx:02d}00"

        vol_scale = spot_mid * realized_vol * 0.15
        if vol_scale < 1e-6:
            vol_scale = 100.0

        for i in range(n):
            logit = np.clip((spot_prices[i] - strike) / vol_scale, -5, 5)
            prob = 1.0 / (1.0 + np.exp(-logit))
            yes_price = np.clip(prob * 98.0 + 1.0 + rng.normal(0, 0.3), 1, 99)
            rows.append({
                "timestamp": timestamps[i],
                "ticker": ticker,
                "yes_price": float(yes_price),
            })

    return pl.DataFrame(rows).sort(["ticker", "timestamp"])


def _build_hp_panel(df: pl.DataFrame) -> pl.DataFrame | None:
    """Long-format -> wide price panel (time_bucket x tickers)."""
    wide = (
        df.select(["timestamp", "ticker", "yes_price"])
        .pivot(on="ticker", index="timestamp", values="yes_price")
        .sort("timestamp")
        .rename({"timestamp": "time_bucket"})
    )
    for col in wide.columns:
        if col != "time_bucket":
            wide = wide.with_columns(pl.col(col).forward_fill())
    wide = wide.drop_nulls()
    return wide if wide.height >= 10 else None


# ---------------------------------------------------------------------------
# Walk-forward collection with signal markers
# ---------------------------------------------------------------------------

def walk_forward_with_signals(
    df: pl.DataFrame,
    tickers: list[str],
    window: int = 60,
    max_steps: int = 150,
) -> dict:
    """Collect one-step forecast vs. realised cycle, plus simplified entry/exit
    markers based on the z-score threshold logic.
    """
    cfg = PipelineConfig.from_env()
    engine = make_engine(cfg)

    # Per-ticker sorted data
    ticker_data: dict[str, pl.DataFrame] = {}
    for t in tickers:
        td = df.filter(pl.col("ticker") == t).sort("timestamp")
        if td.height > window + 10:
            ticker_data[t] = td

    if len(ticker_data) < 2:
        raise ValueError("Need >= 2 tickers with enough data")

    available_steps = min(td.height for td in ticker_data.values()) - window
    n_steps = min(available_steps, max_steps)

    # Target ticker = first ticker (ATM-ish contract, most interesting dynamics)
    plot_ticker = tickers[0]

    # Accumulators
    steps: list[int] = []
    forecast_cycles: list[float] = []
    realised_cycles: list[float] = []
    residual_stds: list[float] = []

    # Signal markers: (step_index, type)  type in {"entry_buy", "entry_sell", "exit"}
    markers: list[tuple[int, str, float]] = []

    # Simple position tracker for exit logic
    in_position: str | None = None  # "BUY_YES" or "BUY_NO" or None
    entry_step: int = 0

    for step in range(n_steps):
        # Build window panel from all tickers
        rows: list[dict] = []
        actuals: dict[str, float] = {}
        for t, td in ticker_data.items():
            w = td.slice(step, window)
            nxt = td.slice(step + window, 1)
            if nxt.height == 0:
                continue
            actuals[t] = float(nxt["yes_price"][0])
            for i in range(w.height):
                rows.append({
                    "timestamp": w["timestamp"][i],
                    "ticker": t,
                    "yes_price": float(w["yes_price"][i]),
                })

        if len(actuals) < 2:
            continue

        panel = _build_hp_panel(pl.DataFrame(rows))
        if panel is None:
            continue

        # Fit + forecast
        try:
            forecasts = engine.fit_and_forecast(panel)
        except Exception as e:
            logger.warning("Step %d failed: %s", step, e)
            continue

        if plot_ticker not in forecasts or plot_ticker not in actuals:
            continue

        fc = forecasts[plot_ticker]
        actual_price = actuals[plot_ticker]
        # Realised cycle at t+1 approximated as actual_price - trend(t)
        realised_cycle = actual_price - fc.trend

        steps.append(step)
        forecast_cycles.append(fc.forecast_cycle)
        realised_cycles.append(realised_cycle)
        residual_stds.append(fc.residual_std)

        # --- Simplified signal detection (mirrors signal_gen.py z-score logic) ---
        z_score = -fc.current_cycle / fc.residual_std if fc.residual_std > 1e-10 else 0.0
        threshold = cfg.signal_threshold

        if in_position is None:
            # Entry: z-score exceeds threshold and forecast supports mean reversion
            if abs(z_score) > threshold:
                # Forecast swing check (filter 7): reject if diverging or overshooting
                crosses_zero = (fc.current_cycle * fc.forecast_cycle) < 0
                if fc.current_cycle < 0:
                    is_diverging = fc.forecast_cycle < fc.current_cycle
                else:
                    is_diverging = fc.forecast_cycle > fc.current_cycle

                if not crosses_zero and not is_diverging:
                    if z_score > 0:
                        in_position = "BUY_YES"
                        markers.append((step, "entry_buy", realised_cycle))
                    else:
                        in_position = "BUY_NO"
                        markers.append((step, "entry_sell", realised_cycle))
                    entry_step = step
        else:
            # Exit: z-score reverts past exit threshold, or held too long
            exit_z = cfg.exit_z_threshold
            held = step - entry_step
            should_exit = False

            if in_position == "BUY_YES" and z_score < exit_z:
                should_exit = True
            elif in_position == "BUY_NO" and z_score > -exit_z:
                should_exit = True
            elif held >= cfg.holding_period_minutes:
                should_exit = True

            if should_exit:
                markers.append((step, "exit", realised_cycle))
                in_position = None

    return {
        "ticker": plot_ticker,
        "steps": steps,
        "forecast_cycles": forecast_cycles,
        "realised_cycles": realised_cycles,
        "residual_stds": residual_stds,
        "markers": markers,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure2(results: dict, save_path: str = "output/figure2_forecast_vs_realised.png") -> None:
    steps = np.array(results["steps"])
    forecast = np.array(results["forecast_cycles"])
    realised = np.array(results["realised_cycles"])
    sigma = np.array(results["residual_stds"])
    markers = results["markers"]

    # ---- Metrics ----
    errors = forecast - realised
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    # Directional accuracy: did the forecast correctly predict cycle sign?
    dir_correct = np.sum((forecast > 0) == (realised > 0))
    dir_acc = dir_correct / len(steps) if len(steps) > 0 else 0.0

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(14, 6))

    # +/- 1 sigma band around the forecast
    ax.fill_between(
        steps,
        forecast - sigma,
        forecast + sigma,
        alpha=0.18, color="#1f77b4",
        label="$\\pm 1\\,\\sigma_{\\mathrm{cycle}}$",
    )

    # Realised cycle
    ax.plot(
        steps, realised,
        color="black", linewidth=0.9, alpha=0.85,
        label="Realised cycle",
    )

    # Forecast cycle
    ax.plot(
        steps, forecast,
        color="#1f77b4", linewidth=1.4, linestyle="--",
        label="1-step forecast",
    )

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)

    # ---- Entry / exit markers ----
    for step_idx, mtype, cycle_val in markers:
        if mtype == "entry_buy":
            ax.plot(
                step_idx, cycle_val,
                marker="^", color="#2ca02c", markersize=10,
                markeredgecolor="black", markeredgewidth=0.6,
                zorder=5,
            )
        elif mtype == "entry_sell":
            ax.plot(
                step_idx, cycle_val,
                marker="v", color="#d62728", markersize=10,
                markeredgecolor="black", markeredgewidth=0.6,
                zorder=5,
            )
        elif mtype == "exit":
            ax.plot(
                step_idx, cycle_val,
                marker="x", color="#ff7f0e", markersize=9,
                markeredgewidth=2.0,
                zorder=5,
            )

    # Legend entries for markers (plot invisible points for clean legend)
    ax.plot([], [], marker="^", color="#2ca02c", linestyle="None",
            markersize=8, markeredgecolor="black", markeredgewidth=0.6,
            label="Entry BUY YES")
    ax.plot([], [], marker="v", color="#d62728", linestyle="None",
            markersize=8, markeredgecolor="black", markeredgewidth=0.6,
            label="Entry BUY NO")
    ax.plot([], [], marker="x", color="#ff7f0e", linestyle="None",
            markersize=8, markeredgewidth=2.0,
            label="Exit")

    ax.set_xlabel("Walk-forward step (60 s buckets)", fontsize=11)
    ax.set_ylabel("Cycle component (cents)", fontsize=11)
    ax.set_title(
        f"Figure 2 :  One-step HP-DFM cycle forecast vs. realised  \u2014  {results['ticker']}\n"
        f"Dir. accuracy = {dir_acc:.1%}    Bias = {np.mean(errors):+.3f}c",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

    # Summary
    n_entries = sum(1 for _, m, _ in markers if m.startswith("entry"))
    n_exits = sum(1 for _, m, _ in markers if m == "exit")
    print(f"\n{'=' * 60}")
    print("FIGURE 2 SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Ticker           : {results['ticker']}")
    print(f"  Walk-forward steps: {len(steps)}")
    #print(f"  MAE              : {mae:.3f}c")
    #print(f"  RMSE             : {rmse:.3f}c")
    print(f"  Bias             : {np.mean(errors):+.3f}c")
    print(f"  Dir. accuracy    : {dir_acc:.1%}")
    print(f"  Mean sigma_cycle : {np.mean(sigma):.3f}c")
    print(f"  Entry signals    : {n_entries}")
    print(f"  Exit signals     : {n_exits}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Figure 2: HP-DFM one-step forecast vs. realised cycle")
    print("=" * 60)

    # Fetch real BTC 1-min candles (5 hours = 300 candles)
    print("\nFetching BTC spot data ...")
    spot_df = await fetch_spot_klines_with_fallback(limit=300)
    print(f"  {spot_df.height} candles retrieved")

    if spot_df.height < 100:
        print("ERROR: not enough data.")
        return

    # Derive 3 synthetic contracts from spot dynamics
    n_contracts = 3
    print(f"\nDeriving {n_contracts} synthetic contracts ...")
    contracts_df = _spot_to_contract_prices(spot_df, n_contracts=n_contracts)
    tickers = contracts_df["ticker"].unique().to_list()
    print(f"  {contracts_df.height} rows, {len(tickers)} tickers: {tickers}")

    # Walk-forward with signal detection
    print("\nRunning walk-forward analysis with signal detection ...")
    results = walk_forward_with_signals(
        contracts_df, tickers, window=60, max_steps=150,
    )

    # Plot
    print("\nGenerating figure ...")
    plot_figure2(results)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
