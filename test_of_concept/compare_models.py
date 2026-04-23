"""Head-to-head comparison: TFT vs HP-DFM-RTE on identical synthetic data.

Generates synthetic Kalshi BTC price data, runs both models, and compares:
  - Forecast accuracy (MAE, RMSE, directional accuracy)
  - Signal quality (edge magnitude, filter pass rate)
  - Trading profitability (net EV after fees)
  - Latency (wall-clock inference time)

Usage:
    python test_of_concept/compare_models.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.tft.data_fetcher import generate_synthetic_data
from model.tft.feature_engineer import engineer_features
from model.hp_dfm_rte.config import PipelineConfig
from model.hp_dfm_rte.model_engine import HPCycleDFMEngine, make_engine, TickerForecast
from model.hp_dfm_rte.signal_gen import SignalGenerator
from model.hp_dfm_rte.accumulator import TradeAccumulator
from model.hp_dfm_rte.fees import round_trip_fee

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_hp_dfm_panel(df: pl.DataFrame, tickers: list[str]) -> pl.DataFrame | None:
    """Convert long-format data into wide price panel for HP-DFM-RTE.

    The engine expects a Polars DataFrame with columns:
        time_bucket, ticker_1, ticker_2, ...  (wide format, yes_price in cents)
    """
    # Pivot: each ticker becomes a column, values are yes_price
    wide = (
        df.select(["timestamp", "ticker", "yes_price"])
        .pivot(on="ticker", index="timestamp", values="yes_price")
        .sort("timestamp")
    )
    # Rename timestamp -> time_bucket (expected by engine)
    wide = wide.rename({"timestamp": "time_bucket"})

    # Forward-fill then drop nulls
    for col in wide.columns:
        if col != "time_bucket":
            wide = wide.with_columns(pl.col(col).forward_fill())
    wide = wide.drop_nulls()

    if wide.height < 10:
        return None
    return wide


def _run_hp_dfm_rte(df: pl.DataFrame, tickers: list[str]) -> dict:
    """Run HP-DFM-RTE model on the data and collect predictions.

    Returns dict with per-ticker forecasts, timing, and signal stats.
    """
    cfg = PipelineConfig()

    # Build wide panel
    panel = _build_hp_dfm_panel(df, tickers)
    if panel is None:
        return {"error": "Not enough data for HP-DFM-RTE panel"}

    engine = make_engine(cfg)
    signal_gen = SignalGenerator(cfg)

    # Run fit + forecast (timed)
    t0 = time.perf_counter()
    forecasts = engine.fit_and_forecast(panel)
    fit_time_ms = (time.perf_counter() - t0) * 1000.0

    # Build current prices dict for signal gen
    prices = {}
    for t in tickers:
        col = t
        if col in panel.columns:
            prices[t] = float(panel[col][-1])

    # Run signal evaluation
    signals = signal_gen.evaluate(forecasts, prices=prices)

    return {
        "forecasts": forecasts,
        "signals": signals,
        "filter_stats": signal_gen.get_filter_stats(),
        "fit_time_ms": fit_time_ms,
        "prices": prices,
    }


def _run_tft(df: pl.DataFrame) -> dict:
    """Run TFT model on the data and collect predictions.

    Loads the trained checkpoint if available. Falls back to passthrough.
    """
    ckpt_path = Path(__file__).resolve().parent.parent / "model" / "tft" / "checkpoints" / "tft_model"
    ckpt_file = ckpt_path.with_suffix(".ckpt")

    if not ckpt_file.exists():
        return {"error": f"No TFT checkpoint at {ckpt_file}"}

    try:
        from model.tft.tft_model import TFTPredictor
    except ImportError as e:
        return {"error": f"TFT dependencies missing: {e}"}

    # Feature engineering
    features_df = engineer_features(df)

    # Load model and predict (timed)
    t0 = time.perf_counter()
    predictor = TFTPredictor.load(ckpt_path)
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    predictions = predictor.predict(features_df)
    predict_time_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "predictions": predictions,
        "load_time_ms": load_time_ms,
        "predict_time_ms": predict_time_ms,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def _evaluate_hp_dfm_rte_accuracy(
    df: pl.DataFrame,
    tickers: list[str],
) -> dict:
    """Walk-forward evaluation of HP-DFM-RTE: fit on window, predict next step."""
    cfg = PipelineConfig()
    engine = make_engine(cfg)

    window_size = 60  # observations per ticker
    step_size = 1
    errors = []         # signed: predicted - actual
    abs_errors = []
    direction_correct = 0
    direction_total = 0
    fit_times = []

    # Get per-ticker sorted data
    ticker_data = {}
    for t in tickers:
        td = df.filter(pl.col("ticker") == t).sort("timestamp")
        if td.height > window_size + 10:
            ticker_data[t] = td

    if len(ticker_data) < 2:
        return {"error": "Need >= 2 tickers with enough data"}

    # Walk forward through the data
    max_steps = min(len(td) for td in ticker_data.values()) - window_size
    max_steps = min(max_steps, 100)  # cap for speed

    for step in range(0, max_steps, step_size):
        # Build wide panel from window
        rows = []
        actuals = {}
        for t, td in ticker_data.items():
            window = td.slice(step, window_size)
            next_row = td.slice(step + window_size, 1)
            if next_row.height == 0:
                continue
            actuals[t] = float(next_row["yes_price"][0])
            for i in range(window.height):
                rows.append({
                    "timestamp": window["timestamp"][i],
                    "ticker": t,
                    "yes_price": float(window["yes_price"][i]),
                })

        if len(actuals) < 2:
            continue

        window_df = pl.DataFrame(rows)
        panel = _build_hp_dfm_panel(window_df, list(actuals.keys()))
        if panel is None or panel.height < 10:
            continue

        # Fit and forecast
        t0 = time.perf_counter()
        try:
            forecasts = engine.fit_and_forecast(panel)
        except Exception:
            continue
        fit_times.append((time.perf_counter() - t0) * 1000.0)

        # Evaluate: predicted = trend + forecast_cycle
        for t, fc in forecasts.items():
            if t not in actuals:
                continue
            predicted = fc.trend + fc.forecast_cycle
            actual = actuals[t]
            current = fc.trend + fc.current_cycle

            err = predicted - actual
            errors.append(err)
            abs_errors.append(abs(err))

            # Direction: did we correctly predict up/down from current?
            pred_dir = predicted - current
            actual_dir = actual - current
            if pred_dir != 0 and actual_dir != 0:
                direction_total += 1
                if (pred_dir > 0) == (actual_dir > 0):
                    direction_correct += 1

    if not abs_errors:
        return {"error": "No valid evaluation steps"}

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        "bias": float(np.mean(errors)),
        "median_ae": float(np.median(abs_errors)),
        "direction_accuracy": direction_correct / direction_total if direction_total > 0 else 0.0,
        "direction_total": direction_total,
        "n_predictions": len(errors),
        "mean_fit_time_ms": float(np.mean(fit_times)),
        "p90_fit_time_ms": float(np.percentile(fit_times, 90)),
    }


def _evaluate_tft_accuracy(
    df: pl.DataFrame,
    predictions: list,
) -> dict:
    """Evaluate TFT predictions against actual last prices."""
    if not predictions:
        return {"error": "No TFT predictions"}

    errors = []
    abs_errors = []
    direction_correct = 0
    direction_total = 0
    uncertainties = []
    ci_hits = 0

    for pred in predictions:
        actual = pred.current_price  # latest observed
        predicted = pred.predicted_price
        err = predicted - actual
        errors.append(err)
        abs_errors.append(abs(err))
        uncertainties.append(pred.uncertainty)

        # CI coverage: does actual fall within [lower, upper]?
        if pred.lower_bound <= actual <= pred.upper_bound:
            ci_hits += 1

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.mean(errors)),
        "median_ae": float(np.median(abs_errors)),
        "n_predictions": len(errors),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "ci_coverage_90": ci_hits / len(predictions) if predictions else 0.0,
    }


# ── Signal quality comparison ────────────────────────────────────────────────

def _compare_signals(hp_result: dict, tft_result: dict) -> dict:
    """Compare signal quality between the two models."""
    comparison = {}

    # HP-DFM-RTE signals
    if "signals" in hp_result:
        signals = hp_result["signals"]
        comparison["hp_dfm_rte"] = {
            "n_signals": len(signals),
            "filter_stats": hp_result.get("filter_stats", {}),
        }
        if signals:
            z_scores = [abs(s.z_score) for s in signals]
            comparison["hp_dfm_rte"]["mean_z_score"] = float(np.mean(z_scores))
            comparison["hp_dfm_rte"]["max_z_score"] = float(np.max(z_scores))

            # Edge in cents (from forecast)
            edges = []
            for s in signals:
                edge_cents = abs(s.forecast_cycle - s.current_cycle) * 100
                edges.append(edge_cents)
            comparison["hp_dfm_rte"]["mean_edge_cents"] = float(np.mean(edges))

    # TFT predictions -> signals
    if "predictions" in tft_result:
        preds = tft_result["predictions"]
        # Apply same thresholds as main.py
        strong_signals = [p for p in preds if abs(p.deviation) >= 5.0]  # 5c edge
        comparison["tft"] = {
            "n_predictions": len(preds),
            "n_strong_signals": len(strong_signals),
        }
        if preds:
            deviations = [abs(p.deviation) for p in preds]
            comparison["tft"]["mean_edge_cents"] = float(np.mean(deviations))
            comparison["tft"]["max_edge_cents"] = float(np.max(deviations))

            evs = [p.net_ev() for p in preds]
            comparison["tft"]["mean_net_ev"] = float(np.mean(evs))
            comparison["tft"]["pct_positive_ev"] = sum(1 for e in evs if e > 0) / len(evs)

            confidences = [p.confidence for p in preds]
            comparison["tft"]["mean_confidence"] = float(np.mean(confidences))

    return comparison


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  MODEL COMPARISON: TFT vs HP-DFM-RTE")
    print("=" * 70)

    # Generate synthetic data (same for both models)
    n_tickers = 5
    n_steps = 500
    seed = 42
    print(f"\nGenerating synthetic data: {n_tickers} tickers x {n_steps} steps (seed={seed})...")
    df = generate_synthetic_data(n_tickers=n_tickers, n_steps=n_steps, seed=seed)
    tickers = df["ticker"].unique().to_list()
    print(f"  Data shape: {df.height} rows, tickers: {tickers}")

    # ── Run HP-DFM-RTE ──────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  HP-DFM-RTE MODEL")
    print("-" * 70)

    # Walk-forward accuracy
    print("\nRunning walk-forward evaluation (100 steps)...")
    hp_accuracy = _evaluate_hp_dfm_rte_accuracy(df, tickers)
    if "error" in hp_accuracy:
        print(f"  ERROR: {hp_accuracy['error']}")
    else:
        print(f"  MAE:                {hp_accuracy['mae']:.3f}c")
        print(f"  RMSE:               {hp_accuracy['rmse']:.3f}c")
        print(f"  Bias:               {hp_accuracy['bias']:+.3f}c")
        print(f"  Median AE:          {hp_accuracy['median_ae']:.3f}c")
        print(f"  Direction Accuracy: {hp_accuracy['direction_accuracy']:.1%} ({hp_accuracy['direction_total']} samples)")
        print(f"  N predictions:      {hp_accuracy['n_predictions']}")
        print(f"  Mean fit time:      {hp_accuracy['mean_fit_time_ms']:.1f} ms")
        print(f"  P90 fit time:       {hp_accuracy['p90_fit_time_ms']:.1f} ms")

    # Single-shot signal evaluation
    print("\nRunning signal generation on full dataset...")
    hp_result = _run_hp_dfm_rte(df, tickers)
    if "error" in hp_result:
        print(f"  ERROR: {hp_result['error']}")
    else:
        print(f"  Fit time:           {hp_result['fit_time_ms']:.1f} ms")
        print(f"  Signals generated:  {len(hp_result['signals'])}")
        print(f"  Filter stats:       {hp_result['filter_stats']}")

        if hp_result["forecasts"]:
            print("\n  Per-ticker forecasts:")
            for t, fc in sorted(hp_result["forecasts"].items()):
                z = -fc.current_cycle / fc.residual_std
                pred_price = fc.trend + fc.forecast_cycle
                current_price = fc.trend + fc.current_cycle
                print(f"    {t}: current={current_price:.1f}c  pred={pred_price:.1f}c  "
                      f"z={z:+.3f}  cycle={fc.current_cycle:.4f}  std={fc.residual_std:.4f}")

    # ── Run TFT ─────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("  TFT MODEL")
    print("-" * 70)

    tft_result = _run_tft(df)
    if "error" in tft_result:
        print(f"\n  ERROR: {tft_result['error']}")
        print("  (Run `python -m model.tft.train_tft --synthetic` to train first)")
    else:
        print(f"\n  Load time:          {tft_result['load_time_ms']:.1f} ms")
        print(f"  Predict time:       {tft_result['predict_time_ms']:.1f} ms")
        print(f"  Total inference:    {tft_result['load_time_ms'] + tft_result['predict_time_ms']:.1f} ms")

        preds = tft_result["predictions"]
        print(f"  Predictions:        {len(preds)}")

        tft_accuracy = _evaluate_tft_accuracy(df, preds)
        if "error" not in tft_accuracy:
            print(f"  MAE:                {tft_accuracy['mae']:.3f}c")
            print(f"  RMSE:               {tft_accuracy['rmse']:.3f}c")
            print(f"  Bias:               {tft_accuracy['bias']:+.3f}c")
            print(f"  Median AE:          {tft_accuracy['median_ae']:.3f}c")
            print(f"  Mean uncertainty:   {tft_accuracy['mean_uncertainty']:.3f}c")
            print(f"  90% CI coverage:    {tft_accuracy['ci_coverage_90']:.1%}")

        print("\n  Per-ticker predictions:")
        for pred in sorted(preds, key=lambda p: p.ticker):
            print(f"    {pred.ticker}: current={pred.current_price:.1f}c  pred={pred.predicted_price:.1f}c  "
                  f"dev={pred.deviation:+.1f}c  conf={pred.confidence:.2f}  "
                  f"CI=[{pred.lower_bound:.1f}, {pred.upper_bound:.1f}]  "
                  f"EV={pred.net_ev():+.2f}c")

    # ── Head-to-head comparison ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    signal_comparison = _compare_signals(hp_result, tft_result)

    # Accuracy comparison table
    print("\n  ACCURACY:")
    print(f"  {'Metric':<25} {'HP-DFM-RTE':>12} {'TFT':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")

    hp_mae = hp_accuracy.get("mae", float("nan"))
    hp_rmse = hp_accuracy.get("rmse", float("nan"))
    hp_bias = hp_accuracy.get("bias", float("nan"))
    hp_dir = hp_accuracy.get("direction_accuracy", float("nan"))

    tft_mae = tft_accuracy.get("mae", float("nan")) if "error" not in tft_result else float("nan")
    tft_rmse = tft_accuracy.get("rmse", float("nan")) if "error" not in tft_result else float("nan")
    tft_bias = tft_accuracy.get("bias", float("nan")) if "error" not in tft_result else float("nan")

    print(f"  {'MAE (cents)':<25} {hp_mae:>12.3f} {tft_mae:>12.3f}")
    print(f"  {'RMSE (cents)':<25} {hp_rmse:>12.3f} {tft_rmse:>12.3f}")
    print(f"  {'Bias (cents)':<25} {hp_bias:>+12.3f} {tft_bias:>+12.3f}")

    if "error" not in hp_accuracy:
        print(f"  {'Direction Accuracy':<25} {hp_dir:>11.1%} {'N/A':>12}")

    # Timing comparison
    print("\n  LATENCY:")
    hp_time = hp_accuracy.get("mean_fit_time_ms", hp_result.get("fit_time_ms", float("nan")))
    tft_time = (tft_result.get("predict_time_ms", float("nan"))
                if "error" not in tft_result else float("nan"))
    print(f"  {'Inference (ms)':<25} {hp_time:>12.1f} {tft_time:>12.1f}")

    # Signal comparison
    print("\n  SIGNAL QUALITY:")
    if "hp_dfm_rte" in signal_comparison:
        hp_sig = signal_comparison["hp_dfm_rte"]
        print(f"  HP-DFM-RTE: {hp_sig['n_signals']} signals, "
              f"mean |z|={hp_sig.get('mean_z_score', 0):.3f}, "
              f"mean edge={hp_sig.get('mean_edge_cents', 0):.2f}c")
    if "tft" in signal_comparison:
        tft_sig = signal_comparison["tft"]
        print(f"  TFT:        {tft_sig['n_predictions']} predictions, "
              f"{tft_sig['n_strong_signals']} strong signals (>=5c edge), "
              f"mean edge={tft_sig.get('mean_edge_cents', 0):.2f}c")
        print(f"              mean EV={tft_sig.get('mean_net_ev', 0):+.2f}c, "
              f"{tft_sig.get('pct_positive_ev', 0):.0%} positive EV, "
              f"mean confidence={tft_sig.get('mean_confidence', 0):.2f}")

    # Winner summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if not np.isnan(hp_mae) and not np.isnan(tft_mae):
        mae_winner = "HP-DFM-RTE" if hp_mae < tft_mae else "TFT"
        print(f"  MAE winner:       {mae_winner} ({min(hp_mae, tft_mae):.3f}c vs {max(hp_mae, tft_mae):.3f}c)")

    if not np.isnan(hp_rmse) and not np.isnan(tft_rmse):
        rmse_winner = "HP-DFM-RTE" if hp_rmse < tft_rmse else "TFT"
        print(f"  RMSE winner:      {rmse_winner} ({min(hp_rmse, tft_rmse):.3f}c vs {max(hp_rmse, tft_rmse):.3f}c)")

    if not np.isnan(hp_time) and not np.isnan(tft_time):
        speed_winner = "HP-DFM-RTE" if hp_time < tft_time else "TFT"
        print(f"  Speed winner:     {speed_winner} ({min(hp_time, tft_time):.1f} ms vs {max(hp_time, tft_time):.1f} ms)")

    print()


if __name__ == "__main__":
    main()