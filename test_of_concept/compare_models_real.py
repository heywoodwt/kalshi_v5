"""Head-to-head comparison: TFT vs HP-DFM-RTE on REAL market data.

Fetches live BTC spot data from exchange APIs (Coinbase/Kraken/Bybit/Binance),
derives realistic Kalshi-style contract prices from spot movements, and
runs both models in a walk-forward backtest.

Evaluation:
  - Walk-forward 1-step-ahead accuracy (MAE, RMSE, directional accuracy)
  - Signal quality (edge, filter pass rate, EV after fees)
  - Latency (wall-clock fit/inference time)
  - Simulated PnL from signals generated

Usage:
    python test_of_concept/compare_models_real.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.tft.data_fetcher import (
    fetch_spot_klines_with_fallback,
    fetch_funding_rate_with_fallback,
    fetch_open_interest_with_fallback,
)
from model.tft.feature_engineer import engineer_features
from model.hp_dfm_rte.config import PipelineConfig
from model.hp_dfm_rte.model_engine import make_engine
from model.hp_dfm_rte.signal_gen import SignalGenerator
from model.hp_dfm_rte.fees import round_trip_fee

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data Pipeline ────────────────────────────────────────────────────────────

def _spot_to_contract_prices(
    spot_df: pl.DataFrame,
    n_contracts: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """Derive realistic Kalshi BTC contract prices from real spot data.

    Each contract represents a binary option on BTC being above/below a
    strike price at a future time. Contract yes_price (0-99c) is computed
    from distance-to-strike normalized by realized volatility.

    This uses REAL spot price dynamics -- the only synthetic element is
    the mapping from spot -> contract price, which mirrors how Kalshi
    contracts actually behave.
    """
    rng = np.random.default_rng(seed)
    spot_prices = spot_df["close"].to_numpy()
    timestamps = spot_df["timestamp"].to_list()
    n = len(spot_prices)

    # Compute realized volatility for strike spacing
    returns = np.diff(np.log(spot_prices))
    realized_vol = np.std(returns) * np.sqrt(len(returns))
    spot_mid = np.median(spot_prices)

    rows = []
    for c_idx in range(n_contracts):
        # Strike prices spaced around the median spot
        # Inner contracts near 50c, outer contracts near edges
        strike_offset = (c_idx - n_contracts // 2) * spot_mid * realized_vol * 0.3
        strike = spot_mid + strike_offset

        # Ticker format matches Kalshi BTC contracts
        hour = 17 + c_idx
        ticker = f"KXBTCD-26APR{hour:02d}00"

        for i in range(n):
            # Binary option price: probability spot > strike at expiry
            # Approximated via logistic function of (spot - strike) / vol_scale
            vol_scale = spot_mid * realized_vol * 0.15
            if vol_scale < 1e-6:
                vol_scale = 100.0

            logit = (spot_prices[i] - strike) / vol_scale
            # Clip logit to avoid extreme prices
            logit = np.clip(logit, -5, 5)
            prob = 1.0 / (1.0 + np.exp(-logit))

            # Convert to cents (1-99 range) with small noise for realism
            yes_price = prob * 98.0 + 1.0  # [1, 99]
            noise = rng.normal(0, 0.3)  # Small microstructure noise
            yes_price = np.clip(yes_price + noise, 1.0, 99.0)

            rows.append({
                "timestamp": timestamps[i],
                "ticker": ticker,
                "yes_price": float(yes_price),
                "volume": float(max(1, rng.poisson(8))),
                "spot_price": float(spot_prices[i]),
            })

    df = pl.DataFrame(rows).sort(["ticker", "timestamp"])
    return df


async def _fetch_real_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Fetch real BTC data from exchange APIs."""
    print("  Fetching real BTC spot data from exchanges...")
    spot_df, funding_df, oi_df = await asyncio.gather(
        fetch_spot_klines_with_fallback(limit=500),
        fetch_funding_rate_with_fallback(),
        fetch_open_interest_with_fallback(),
    )
    return spot_df, funding_df, oi_df


def _enrich_contracts(
    contracts_df: pl.DataFrame,
    funding_df: pl.DataFrame,
    oi_df: pl.DataFrame,
) -> pl.DataFrame:
    """Join funding rate and open interest to contract data."""
    # Join funding rate via asof
    if funding_df.height > 0:
        funding_join = funding_df.select([
            pl.col("timestamp").dt.truncate("1m"),
            pl.col("funding_rate"),
        ]).sort("timestamp")
        contracts_df = (
            contracts_df.sort("timestamp")
            .join_asof(funding_join, on="timestamp", strategy="backward")
        )
    else:
        contracts_df = contracts_df.with_columns(pl.lit(0.0).alias("funding_rate"))

    # Join open interest via asof
    if oi_df.height > 0:
        oi_join = oi_df.select([
            pl.col("timestamp").dt.truncate("1m"),
            pl.col("open_interest"),
        ]).sort("timestamp")
        contracts_df = (
            contracts_df.sort("timestamp")
            .join_asof(oi_join, on="timestamp", strategy="backward")
        )
    else:
        contracts_df = contracts_df.with_columns(pl.lit(0.0).alias("open_interest"))

    # Fill nulls
    contracts_df = contracts_df.with_columns([
        pl.col("funding_rate").forward_fill().fill_null(0.0),
        pl.col("open_interest").forward_fill().fill_null(0.0),
    ])

    return contracts_df.sort(["ticker", "timestamp"])


# ── Model Runners ────────────────────────────────────────────────────────────

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


def _walk_forward_hp_dfm_rte(
    df: pl.DataFrame,
    tickers: list[str],
    window: int = 60,
    max_steps: int = 200,
) -> dict:
    """Walk-forward backtest for HP-DFM-RTE.

    At each step: fit on [t-window : t], predict t+1, compare to actual.
    Also generates signals and simulates PnL.
    """
    cfg = PipelineConfig()
    engine = make_engine(cfg)
    signal_gen = SignalGenerator(cfg)

    # Per-ticker sorted arrays
    ticker_data = {}
    for t in tickers:
        td = df.filter(pl.col("ticker") == t).sort("timestamp")
        if td.height > window + 10:
            ticker_data[t] = td

    if len(ticker_data) < 2:
        return {"error": "Need >= 2 tickers with enough data"}

    available_steps = min(td.height for td in ticker_data.values()) - window
    n_steps = min(available_steps, max_steps)

    # Accumulators
    errors = []
    abs_errors = []
    dir_correct = 0
    dir_total = 0
    fit_times = []
    all_signals = []
    simulated_pnl = []  # Per-signal simulated PnL

    for step in range(n_steps):
        # Build window panel
        rows = []
        actuals = {}
        current_prices = {}
        for t, td in ticker_data.items():
            w = td.slice(step, window)
            nxt = td.slice(step + window, 1)
            if nxt.height == 0:
                continue
            actuals[t] = float(nxt["yes_price"][0])
            current_prices[t] = float(w["yes_price"][-1])
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
        t0 = time.perf_counter()
        try:
            forecasts = engine.fit_and_forecast(panel)
        except Exception:
            continue
        fit_ms = (time.perf_counter() - t0) * 1000.0
        fit_times.append(fit_ms)

        # Accuracy evaluation
        for t, fc in forecasts.items():
            if t not in actuals:
                continue
            predicted = fc.trend + fc.forecast_cycle
            actual = actuals[t]
            current = fc.trend + fc.current_cycle

            err = predicted - actual
            errors.append(err)
            abs_errors.append(abs(err))

            pred_dir = predicted - current
            actual_dir = actual - current
            if pred_dir != 0 and actual_dir != 0:
                dir_total += 1
                if (pred_dir > 0) == (actual_dir > 0):
                    dir_correct += 1

        # Signal evaluation (every 5 steps to avoid cooldown blocking)
        if step % 5 == 0:
            signals = signal_gen.evaluate(forecasts, prices=current_prices)
            for sig in signals:
                all_signals.append(sig)
                # Simulate: did the actual price move in the signal direction?
                t = sig.ticker
                if t in actuals and t in current_prices:
                    actual_move = actuals[t] - current_prices[t]
                    if sig.direction.value == "BUY_YES":
                        # Bought yes: profit if price went up
                        pnl = actual_move
                    else:
                        # Bought no: profit if price went down
                        pnl = -actual_move
                    # Subtract estimated round-trip fee
                    price_int = max(1, min(99, int(round(current_prices[t]))))
                    fee = round_trip_fee(price_int, price_int, maker=True)
                    simulated_pnl.append(pnl - fee)

    if not abs_errors:
        return {"error": "No valid evaluation steps"}

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        "bias": float(np.mean(errors)),
        "median_ae": float(np.median(abs_errors)),
        "max_ae": float(np.max(abs_errors)),
        "direction_accuracy": dir_correct / dir_total if dir_total > 0 else 0.0,
        "direction_total": dir_total,
        "n_predictions": len(errors),
        "mean_fit_ms": float(np.mean(fit_times)),
        "p90_fit_ms": float(np.percentile(fit_times, 90)),
        "n_signals": len(all_signals),
        "filter_stats": signal_gen.get_filter_stats(),
        "simulated_pnl": simulated_pnl,
        "total_pnl": float(np.sum(simulated_pnl)) if simulated_pnl else 0.0,
        "mean_pnl": float(np.mean(simulated_pnl)) if simulated_pnl else 0.0,
        "win_rate": (sum(1 for p in simulated_pnl if p > 0) / len(simulated_pnl)
                     if simulated_pnl else 0.0),
    }


def _walk_forward_tft(
    df: pl.DataFrame,
    tickers: list[str],
    window: int = 80,
    max_steps: int = 200,
) -> dict:
    """Walk-forward backtest for TFT.

    Loads trained checkpoint, runs inference on sliding windows, evaluates
    predictions against actual next-step prices.
    """
    ckpt_path = (Path(__file__).resolve().parent.parent
                 / "model" / "tft" / "checkpoints" / "tft_model")
    if not ckpt_path.with_suffix(".ckpt").exists():
        return {"error": f"No TFT checkpoint at {ckpt_path.with_suffix('.ckpt')}"}

    try:
        from model.tft.tft_model import TFTPredictor
    except ImportError as e:
        return {"error": f"TFT dependencies missing: {e}"}

    # Load model once
    t0 = time.perf_counter()
    predictor = TFTPredictor.load(ckpt_path)
    load_ms = (time.perf_counter() - t0) * 1000.0

    # Per-ticker sorted data
    ticker_data = {}
    for t in tickers:
        td = df.filter(pl.col("ticker") == t).sort("timestamp")
        if td.height > window + 10:
            ticker_data[t] = td

    if not ticker_data:
        return {"error": "Not enough data for TFT walk-forward"}

    available_steps = min(td.height for td in ticker_data.values()) - window
    # TFT inference is slower, so evaluate at wider intervals
    step_interval = max(1, available_steps // min(max_steps, 50))
    eval_steps = list(range(0, available_steps, step_interval))[:max_steps]

    errors = []
    abs_errors = []
    dir_correct = 0
    dir_total = 0
    predict_times = []
    ci_hits = 0
    uncertainties = []
    simulated_pnl = []

    for step in eval_steps:
        # Build window DataFrame for all tickers
        rows = []
        actuals = {}
        current_prices = {}
        for t, td in ticker_data.items():
            w = td.slice(step, window)
            nxt = td.slice(step + window, 1)
            if nxt.height == 0 or w.height < window:
                continue
            actuals[t] = float(nxt["yes_price"][0])
            current_prices[t] = float(w["yes_price"][-1])
            for i in range(w.height):
                row = {"timestamp": w["timestamp"][i], "ticker": t,
                       "yes_price": float(w["yes_price"][i])}
                # Add optional columns if present
                for col in ["volume", "spot_price", "funding_rate", "open_interest"]:
                    if col in w.columns:
                        row[col] = float(w[col][i])
                rows.append(row)

        if not actuals:
            continue

        window_df = pl.DataFrame(rows)

        # Feature engineering + inference
        try:
            features_df = engineer_features(window_df)
            if features_df.height < 70:
                continue

            t0 = time.perf_counter()
            predictions = predictor.predict(features_df)
            predict_ms = (time.perf_counter() - t0) * 1000.0
            predict_times.append(predict_ms)
        except Exception as e:
            logger.debug("TFT inference failed at step %d: %s", step, e)
            continue

        # Evaluate each prediction
        for pred in predictions:
            t = pred.ticker
            if t not in actuals:
                continue
            actual = actuals[t]
            current = current_prices.get(t, pred.current_price)

            err = pred.predicted_price - actual
            errors.append(err)
            abs_errors.append(abs(err))
            uncertainties.append(pred.uncertainty)

            if pred.lower_bound <= actual <= pred.upper_bound:
                ci_hits += 1

            # Direction accuracy
            pred_dir = pred.predicted_price - current
            actual_dir = actual - current
            if pred_dir != 0 and actual_dir != 0:
                dir_total += 1
                if (pred_dir > 0) == (actual_dir > 0):
                    dir_correct += 1

            # Simulated PnL from strong signals (>= 5c edge)
            if abs(pred.deviation) >= 5.0:
                if pred.deviation > 0:
                    pnl = actual - current  # Bought yes
                else:
                    pnl = -(actual - current)  # Bought no
                price_int = max(1, min(99, int(round(current))))
                fee = round_trip_fee(price_int, price_int, maker=True)
                simulated_pnl.append(pnl - fee)

    if not abs_errors:
        return {"error": "No valid TFT predictions"}

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.array(errors) ** 2))),
        "bias": float(np.mean(errors)),
        "median_ae": float(np.median(abs_errors)),
        "max_ae": float(np.max(abs_errors)),
        "direction_accuracy": dir_correct / dir_total if dir_total > 0 else 0.0,
        "direction_total": dir_total,
        "n_predictions": len(errors),
        "load_ms": load_ms,
        "mean_predict_ms": float(np.mean(predict_times)) if predict_times else 0.0,
        "p90_predict_ms": float(np.percentile(predict_times, 90)) if predict_times else 0.0,
        "mean_uncertainty": float(np.mean(uncertainties)),
        "ci_coverage_90": ci_hits / len(errors) if errors else 0.0,
        "n_signals": len(simulated_pnl),
        "simulated_pnl": simulated_pnl,
        "total_pnl": float(np.sum(simulated_pnl)) if simulated_pnl else 0.0,
        "mean_pnl": float(np.mean(simulated_pnl)) if simulated_pnl else 0.0,
        "win_rate": (sum(1 for p in simulated_pnl if p > 0) / len(simulated_pnl)
                     if simulated_pnl else 0.0),
    }


# ── Display ──────────────────────────────────────────────────────────────────

def _print_section(title: str, char: str = "-"):
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}")


def _print_model_results(name: str, r: dict):
    """Print results for one model."""
    if "error" in r:
        print(f"\n  ERROR: {r['error']}")
        return

    print(f"\n  Accuracy ({r['n_predictions']} predictions, {r['direction_total']} directional):")
    print(f"    MAE:                {r['mae']:.3f}c")
    print(f"    RMSE:               {r['rmse']:.3f}c")
    print(f"    Bias:               {r['bias']:+.3f}c")
    print(f"    Median AE:          {r['median_ae']:.3f}c")
    print(f"    Max AE:             {r['max_ae']:.3f}c")
    print(f"    Direction Accuracy: {r['direction_accuracy']:.1%}")

    # Model-specific metrics
    if "ci_coverage_90" in r:
        print(f"    90% CI Coverage:    {r['ci_coverage_90']:.1%}")
        print(f"    Mean Uncertainty:   {r['mean_uncertainty']:.3f}c")

    # Latency
    if "mean_fit_ms" in r:
        print(f"\n  Latency:")
        print(f"    Mean fit:           {r['mean_fit_ms']:.1f} ms")
        print(f"    P90 fit:            {r['p90_fit_ms']:.1f} ms")
    if "load_ms" in r:
        print(f"\n  Latency:")
        print(f"    Model load:         {r['load_ms']:.1f} ms")
        print(f"    Mean predict:       {r['mean_predict_ms']:.1f} ms")
        print(f"    P90 predict:        {r['p90_predict_ms']:.1f} ms")

    # Signal / PnL
    print(f"\n  Signal Generation:")
    print(f"    Signals:            {r['n_signals']}")
    if "filter_stats" in r:
        fs = r["filter_stats"]
        total_evaluated = sum(fs.values())
        print(f"    Filter stats:       {fs}")
        if total_evaluated > 0:
            print(f"    Pass rate:          {fs.get('passed', 0)}/{total_evaluated} "
                  f"({fs.get('passed', 0)/total_evaluated:.1%})")

    if r["n_signals"] > 0:
        print(f"\n  Simulated Trading ({r['n_signals']} trades):")
        print(f"    Total PnL:          {r['total_pnl']:+.1f}c")
        print(f"    Mean PnL/trade:     {r['mean_pnl']:+.2f}c")
        print(f"    Win rate:           {r['win_rate']:.1%}")
        if r["simulated_pnl"]:
            pnl = r["simulated_pnl"]
            print(f"    Best trade:         {max(pnl):+.2f}c")
            print(f"    Worst trade:        {min(pnl):+.2f}c")
            # Sharpe-like ratio
            if len(pnl) > 1 and np.std(pnl) > 0:
                sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(len(pnl))
                print(f"    Sharpe (annualized): {sharpe:.2f}")


def main():
    print("=" * 70)
    print("  MODEL COMPARISON: TFT vs HP-DFM-RTE (REAL DATA)")
    print("=" * 70)

    # ── Fetch real data ──────────────────────────────────────────────────
    _print_section("DATA PIPELINE", "=")

    print("\n  Fetching real BTC data from exchange APIs...")
    spot_df, funding_df, oi_df = asyncio.run(_fetch_real_data())
    print(f"    Spot:     {spot_df.height} candles "
          f"({spot_df['timestamp'].min()} to {spot_df['timestamp'].max()})")
    print(f"    Funding:  {funding_df.height} records")
    print(f"    OI:       {oi_df.height} records")

    if spot_df.height < 100:
        print("\n  ERROR: Not enough spot data. Check network connection.")
        return

    # Derive contract prices from real spot
    n_contracts = 5
    print(f"\n  Deriving {n_contracts} contract prices from real spot dynamics...")
    contracts_df = _spot_to_contract_prices(spot_df, n_contracts=n_contracts)
    tickers = contracts_df["ticker"].unique().to_list()
    print(f"    Contracts: {contracts_df.height} rows, {len(tickers)} tickers")
    print(f"    Tickers:   {tickers}")

    # Enrich with funding + OI
    contracts_df = _enrich_contracts(contracts_df, funding_df, oi_df)

    # Print sample price ranges
    print("\n  Contract price summary:")
    for t in sorted(tickers):
        td = contracts_df.filter(pl.col("ticker") == t)
        prices = td["yes_price"]
        print(f"    {t}: min={prices.min():.1f}c  max={prices.max():.1f}c  "
              f"mean={prices.mean():.1f}c  std={prices.std():.2f}c  n={td.height}")

    # Spot price summary
    print(f"\n  BTC spot: ${spot_df['close'].min():.0f} - ${spot_df['close'].max():.0f} "
          f"(range: ${spot_df['close'].max() - spot_df['close'].min():.0f})")

    # ── HP-DFM-RTE ───────────────────────────────────────────────────────
    _print_section("HP-DFM-RTE MODEL")
    print("\n  Running walk-forward backtest (200 steps, window=60)...")

    hp_result = _walk_forward_hp_dfm_rte(contracts_df, tickers, window=60, max_steps=200)
    _print_model_results("HP-DFM-RTE", hp_result)

    # ── TFT ──────────────────────────────────────────────────────────────
    _print_section("TFT MODEL")
    print("\n  Running walk-forward backtest (50 steps, window=80)...")

    tft_result = _walk_forward_tft(contracts_df, tickers, window=80, max_steps=50)
    _print_model_results("TFT", tft_result)

    # ── Head-to-head ─────────────────────────────────────────────────────
    _print_section("HEAD-TO-HEAD COMPARISON", "=")

    hp_ok = "error" not in hp_result
    tft_ok = "error" not in tft_result

    if hp_ok and tft_ok:
        print(f"\n  {'Metric':<25} {'HP-DFM-RTE':>14} {'TFT':>14} {'Winner':>14}")
        print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*14}")

        metrics = [
            ("MAE (cents)", "mae", False),
            ("RMSE (cents)", "rmse", False),
            ("Bias (cents)", "bias", None),  # None = closer to 0 wins
            ("Median AE (cents)", "median_ae", False),
            ("Direction Accuracy", "direction_accuracy", True),
        ]

        for label, key, higher_better in metrics:
            hp_val = hp_result[key]
            tft_val = tft_result[key]

            if higher_better is None:
                # Closer to 0 wins
                winner = "HP-DFM-RTE" if abs(hp_val) < abs(tft_val) else "TFT"
                hp_fmt = f"{hp_val:+.3f}"
                tft_fmt = f"{tft_val:+.3f}"
            elif key == "direction_accuracy":
                winner = "HP-DFM-RTE" if hp_val > tft_val else "TFT"
                hp_fmt = f"{hp_val:.1%}"
                tft_fmt = f"{tft_val:.1%}"
            elif higher_better:
                winner = "HP-DFM-RTE" if hp_val > tft_val else "TFT"
                hp_fmt = f"{hp_val:.3f}"
                tft_fmt = f"{tft_val:.3f}"
            else:
                winner = "HP-DFM-RTE" if hp_val < tft_val else "TFT"
                hp_fmt = f"{hp_val:.3f}"
                tft_fmt = f"{tft_val:.3f}"

            if hp_val == tft_val:
                winner = "TIE"

            print(f"  {label:<25} {hp_fmt:>14} {tft_fmt:>14} {winner:>14}")

        # Latency
        hp_latency = hp_result.get("mean_fit_ms", 0)
        tft_latency = tft_result.get("mean_predict_ms", 0)
        lat_winner = "HP-DFM-RTE" if hp_latency < tft_latency else "TFT"
        print(f"  {'Inference (ms)':<25} {hp_latency:>13.1f} {tft_latency:>13.1f}  {lat_winner:>13}")

        # PnL
        if hp_result["n_signals"] > 0 or tft_result["n_signals"] > 0:
            print(f"\n  {'Trading Metric':<25} {'HP-DFM-RTE':>14} {'TFT':>14} {'Winner':>14}")
            print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*14}")

            hp_pnl = hp_result["total_pnl"]
            tft_pnl = tft_result["total_pnl"]
            pnl_winner = "HP-DFM-RTE" if hp_pnl > tft_pnl else "TFT"
            print(f"  {'Total PnL (cents)':<25} {hp_pnl:>+13.1f} {tft_pnl:>+13.1f}  {pnl_winner:>13}")

            hp_wr = hp_result["win_rate"]
            tft_wr = tft_result["win_rate"]
            wr_winner = "HP-DFM-RTE" if hp_wr > tft_wr else "TFT"
            print(f"  {'Win Rate':<25} {hp_wr:>13.1%} {tft_wr:>13.1%}  {wr_winner:>13}")

            print(f"  {'Trades Generated':<25} {hp_result['n_signals']:>14} {tft_result['n_signals']:>14}")

    elif hp_ok:
        print("\n  Only HP-DFM-RTE produced results (TFT failed).")
    elif tft_ok:
        print("\n  Only TFT produced results (HP-DFM-RTE failed).")
    else:
        print("\n  Both models failed to produce results.")

    # ── Summary ──────────────────────────────────────────────────────────
    _print_section("SUMMARY", "=")

    if hp_ok and tft_ok:
        wins = {"HP-DFM-RTE": 0, "TFT": 0}
        for key, lower in [("mae", True), ("rmse", True), ("median_ae", True),
                           ("direction_accuracy", False)]:
            hp_v, tft_v = hp_result[key], tft_result[key]
            if lower:
                wins["HP-DFM-RTE" if hp_v < tft_v else "TFT"] += 1
            else:
                wins["HP-DFM-RTE" if hp_v > tft_v else "TFT"] += 1
        # Bias (closer to 0)
        if abs(hp_result["bias"]) < abs(tft_result["bias"]):
            wins["HP-DFM-RTE"] += 1
        else:
            wins["TFT"] += 1

        overall = "HP-DFM-RTE" if wins["HP-DFM-RTE"] > wins["TFT"] else "TFT"
        print(f"\n  Accuracy metrics won:  HP-DFM-RTE={wins['HP-DFM-RTE']}  TFT={wins['TFT']}")
        print(f"  Overall winner:       {overall}")

        if hp_result["n_signals"] > 0 and tft_result["n_signals"] > 0:
            if hp_result["total_pnl"] > tft_result["total_pnl"]:
                print(f"  More profitable:      HP-DFM-RTE ({hp_result['total_pnl']:+.1f}c vs {tft_result['total_pnl']:+.1f}c)")
            else:
                print(f"  More profitable:      TFT ({tft_result['total_pnl']:+.1f}c vs {hp_result['total_pnl']:+.1f}c)")

    print(f"\n  Data source: Real BTC spot from exchange APIs")
    print(f"  Spot candles: {spot_df.height}")
    print(f"  Contract obs: {contracts_df.height}")
    print()


if __name__ == "__main__":
    main()