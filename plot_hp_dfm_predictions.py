"""Simple plot of HP-DFM-RTE predicted vs. actual cycle values with confidence bands.

Usage:
    python plot_hp_dfm_predictions.py
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


def _spot_to_contract_prices(spot_df: pl.DataFrame, n_contracts: int = 3, seed: int = 42) -> pl.DataFrame:
    """Derive realistic Kalshi BTC contract prices from real spot data."""
    rng = np.random.default_rng(seed)
    spot_prices = spot_df["close"].to_numpy()
    timestamps = spot_df["timestamp"].to_list()
    n = len(spot_prices)

    # Compute realized volatility
    returns = np.diff(np.log(spot_prices))
    realized_vol = np.std(returns) * np.sqrt(len(returns))
    spot_mid = np.median(spot_prices)

    rows = []
    for c_idx in range(n_contracts):
        # Strike prices around median
        strike_offset = (c_idx - n_contracts // 2) * spot_mid * realized_vol * 0.3
        strike = spot_mid + strike_offset
        ticker = f"KXBTCD-26APR{17 + c_idx:02d}00"

        for i in range(n):
            # Binary option price approximation
            vol_scale = spot_mid * realized_vol * 0.15
            if vol_scale < 1e-6:
                vol_scale = 100.0

            logit = (spot_prices[i] - strike) / vol_scale
            logit = np.clip(logit, -5, 5)
            prob = 1.0 / (1.0 + np.exp(-logit))

            # Convert to cents with noise
            yes_price = prob * 98.0 + 1.0
            noise = rng.normal(0, 0.3)
            yes_price = np.clip(yes_price + noise, 1.0, 99.0)

            rows.append({
                "timestamp": timestamps[i],
                "ticker": ticker,
                "yes_price": float(yes_price),
            })

    return pl.DataFrame(rows).sort(["ticker", "timestamp"])


def _build_hp_panel(df: pl.DataFrame) -> pl.DataFrame | None:
    """Convert long-format to wide price panel for HP-DFM-RTE."""
    wide = (
        df.select(["timestamp", "ticker", "yes_price"])
        .pivot(on="ticker", index="timestamp", values="yes_price")
        .sort("timestamp")
    )
    wide = wide.rename({"timestamp": "time_bucket"})
    for col in wide.columns:
        if col != "time_bucket":
            wide = wide.with_columns(pl.col(col).forward_fill())
    wide = wide.drop_nulls()
    return wide if wide.height >= 10 else None


def run_prediction_analysis(
    df: pl.DataFrame,
    tickers: list[str],
    window: int = 60,
    max_steps: int = 100,
) -> dict:
    """Walk-forward prediction collection for HP-DFM-RTE.

    Returns predicted and actual cycle values along with confidence bands.
    """
    cfg = PipelineConfig.from_env()
    engine = make_engine(cfg)

    # Per-ticker sorted arrays
    ticker_data = {}
    for t in tickers:
        td = df.filter(pl.col("ticker") == t).sort("timestamp")
        if td.height > window + 10:
            ticker_data[t] = td

    if len(ticker_data) < 2:
        raise ValueError("Need >= 2 tickers with enough data")

    available_steps = min(td.height for td in ticker_data.values()) - window
    n_steps = min(available_steps, max_steps)

    # Accumulators for plotting - collect first ticker's data for simplicity
    plot_ticker = tickers[0]
    step_indices = []
    predicted_cycles = []
    actual_cycles = []
    current_cycles = []
    confidence_bands = []  # Standard error (residual_std)

    for step in range(n_steps):
        # Build window panel
        rows = []
        actuals = {}
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

        # Collect prediction data for plot_ticker
        if plot_ticker in forecasts and plot_ticker in actuals:
            fc = forecasts[plot_ticker]
            actual_price = actuals[plot_ticker]

            # Actual cycle at next step: actual_price - trend
            # We use current trend as proxy since we don't know future trend
            actual_cycle = actual_price - fc.trend

            step_indices.append(step)
            predicted_cycles.append(fc.forecast_cycle)
            actual_cycles.append(actual_cycle)
            current_cycles.append(fc.current_cycle)
            confidence_bands.append(fc.residual_std)

    return {
        "ticker": plot_ticker,
        "step_indices": step_indices,
        "predicted_cycles": predicted_cycles,
        "actual_cycles": actual_cycles,
        "current_cycles": current_cycles,
        "confidence_bands": confidence_bands,
    }


def plot_predictions(results: dict, save_path: str = "output/hp_dfm_predictions.png"):
    """Generate plot of predicted vs. actual cycles with confidence bands."""
    steps = results["step_indices"]
    predicted = np.array(results["predicted_cycles"])
    actual = np.array(results["actual_cycles"])
    current = np.array(results["current_cycles"])
    std_err = np.array(results["confidence_bands"])

    # Compute MAE on cycles
    mae_cycle = np.mean(np.abs(predicted - actual))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Predicted vs Actual cycles with confidence band
    ax1.plot(steps, actual, 'o-', label='Actual Cycle', color='black', alpha=0.7, markersize=4)
    ax1.plot(steps, predicted, 's--', label='Predicted Cycle', color='blue', alpha=0.7, markersize=4)
    ax1.fill_between(
        steps,
        predicted - 1.96 * std_err,  # 95% confidence interval
        predicted + 1.96 * std_err,
        alpha=0.2,
        color='blue',
        label='95% Confidence Band'
    )
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Cycle Value (cents)', fontsize=11)
    ax1.set_title(
        f'HP-DFM-RFE: Predicted vs Actual Cycles ({results["ticker"]})\n'
        f'MAE = {mae_cycle:.3f}c',
        fontsize=12,
        fontweight='bold'
    )
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Prediction errors
    errors = predicted - actual
    ax2.bar(steps, errors, alpha=0.6, color='red', label='Prediction Error')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1)
    ax2.fill_between(
        steps,
        -1.96 * std_err,
        1.96 * std_err,
        alpha=0.15,
        color='gray',
        label='±1.96σ Band'
    )
    ax2.set_xlabel('Walk-Forward Step', fontsize=11)
    ax2.set_ylabel('Prediction Error (cents)', fontsize=11)
    ax2.set_title('Prediction Errors with Standard Error Bands', fontsize=11)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Ticker:             {results['ticker']}")
    print(f"Steps:              {len(steps)}")
    print(f"MAE (cycles):       {mae_cycle:.3f}c")
    print(f"RMSE (cycles):      {np.sqrt(np.mean(errors**2)):.3f}c")
    print(f"Bias (cycles):      {np.mean(errors):+.3f}c")
    print(f"Mean Std Error:     {np.mean(std_err):.3f}c")
    print(f"Max |error|:        {np.max(np.abs(errors)):.3f}c")

    # Directional accuracy
    pred_dir = predicted - current
    actual_dir = actual - current
    correct_dir = np.sum((pred_dir > 0) == (actual_dir > 0))
    print(f"Direction Accuracy: {correct_dir / len(steps):.1%}")
    print("="*60 + "\n")


async def main():
    """Fetch data, run model, generate plot."""
    print("="*60)
    print("HP-DFM-RTE: Cycle Prediction Plot")
    print("="*60)

    # Fetch real BTC spot data
    print("\nFetching real BTC spot data from exchanges...")
    spot_df = await fetch_spot_klines_with_fallback(limit=300)
    print(f"  Fetched {spot_df.height} candles")

    if spot_df.height < 100:
        print("ERROR: Not enough spot data. Check network connection.")
        return

    # Derive contract prices
    n_contracts = 3
    print(f"\nDeriving {n_contracts} contract prices from spot dynamics...")
    contracts_df = _spot_to_contract_prices(spot_df, n_contracts=n_contracts)
    tickers = contracts_df["ticker"].unique().to_list()
    print(f"  Contracts: {contracts_df.height} rows, {len(tickers)} tickers")

    # Run walk-forward prediction
    print("\nRunning walk-forward prediction analysis...")
    results = run_prediction_analysis(contracts_df, tickers, window=60, max_steps=100)

    # Generate plot
    print("\nGenerating plot...")
    plot_predictions(results)

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())