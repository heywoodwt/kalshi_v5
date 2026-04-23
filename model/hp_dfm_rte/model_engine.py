"""HP-DFM-RTE (HP Filter + Dynamic Factor Model - Real-Time Ensemble)
and EWMA-based cycle detection.

Two engines are available, selected by PipelineConfig.use_ewma:

  EWMAEngine (use_ewma=True):
    - Computes exponentially-weighted mean/std per ticker online
    - cycle = price - EWMA_mean  (deviation from moving average)
    - residual_std = EWMA_std    (adaptive scale; can't overfit)
    - forecast_cycle = current_cycle * (1 - alpha)  (mean-reversion decay)
    - No batch fitting; updates in O(n) per bucket

  HPCycleDFMEngine (use_ewma=False) - HP-DFM-RTE Model:
    1. hpfilter(series, lamb=6.25) per ticker -> cycle + trend
    2. DynamicFactor(cycles, k_factors=2, factor_order=3)
       - k_factors clamped to min(cfg.k_factors, k_endog - 1)
       - enforce_stationarity=True with relaxed fallback
    3. residual_std = empirical cycle std (not in-sample model residuals)
    4. fit(method="bfgs", maxiter=2000) then forecast(steps=1)

Both engines populate TickerForecast.recent_cycles with the last
(momentum_lookback + 1) cycle values for use by the momentum filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

from .config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class TickerForecast:
    ticker: str
    current_cycle: float
    forecast_cycle: float
    trend: float
    residual_std: float
    recent_cycles: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EWMA Engine
# ---------------------------------------------------------------------------

class EWMAEngine:
    """Online exponentially-weighted mean-reversion detector.

    No batch fitting -- each bucket is an O(n_tickers) update.
    Cannot overfit regardless of window length or ticker count.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._last_result: dict[str, TickerForecast] | None = None

    def fit_and_forecast(self, panel: pl.DataFrame) -> dict[str, TickerForecast]:
        """Compute EWMA deviation z-scores from price panel."""
        try:
            return self._compute(panel)
        except Exception:
            logger.exception("EWMA compute failed; returning previous result")
            if self._last_result is not None:
                return self._last_result
            raise

    def _compute(self, panel: pl.DataFrame) -> dict[str, TickerForecast]:
        panel_pd = panel.drop("time_bucket").to_pandas()
        tickers = list(panel_pd.columns)
        alpha = self.cfg.ewma_alpha
        lookback = self.cfg.momentum_lookback + 1

        forecasts: dict[str, TickerForecast] = {}
        for col in tickers:
            series = panel_pd[col].dropna()
            if len(series) < 2:
                continue

            ewm = series.ewm(alpha=alpha, adjust=True)
            ewma_means = ewm.mean()
            ewma_std = float(ewm.std().iloc[-1])

            if ewma_std < 1e-6:
                ewma_std = 1e-6

            # Cycle = deviation from exponentially-weighted mean
            cycles = (series - ewma_means).to_numpy()
            current_cycle = float(cycles[-1])

            # Forecast: one step of EWMA decay back toward the mean
            forecast_cycle = current_cycle * (1.0 - alpha)

            forecasts[col] = TickerForecast(
                ticker=col,
                current_cycle=current_cycle,
                forecast_cycle=forecast_cycle,
                trend=float(ewma_means.iloc[-1]),
                residual_std=ewma_std,
                recent_cycles=list(cycles[-lookback:]),
            )

        self._last_result = forecasts
        logger.info("EWMA compute: %d tickers", len(forecasts))
        return forecasts


# ---------------------------------------------------------------------------
# HP-DFM-RTE: HP Cycle + DFM Real-Time Ensemble Engine
# ---------------------------------------------------------------------------

class HPCycleDFMEngine:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._last_result: dict[str, TickerForecast] | None = None

    def fit_and_forecast(self, panel: pl.DataFrame) -> dict[str, TickerForecast]:
        """Run HP-DFM-RTE on the price panel.

        Returns per-ticker forecasts. Falls back to previous result on failure.
        """
        try:
            return self._fit_and_forecast_inner(panel)
        except Exception:
            logger.exception("DFM fit failed; returning previous result")
            if self._last_result is not None:
                return self._last_result
            raise

    def _fit_and_forecast_inner(self, panel: pl.DataFrame) -> dict[str, TickerForecast]:
        panel_pd = panel.drop("time_bucket").to_pandas()
        tickers = list(panel_pd.columns)
        lookback = self.cfg.momentum_lookback + 1

        # Data quality filter - clip extreme prices and remove outliers
        panel_original = panel_pd.copy()
        outliers_detected = False

        for col in tickers:
            # Clip to valid price range (avoid edge cases at 1c and 99c)
            original_range = (panel_pd[col].min(), panel_pd[col].max())
            panel_pd[col] = panel_pd[col].clip(lower=5, upper=95)

            # Check for and clip outliers (> 3 std devs from mean)
            mean = panel_pd[col].mean()
            std = panel_pd[col].std()
            if std > 0:
                z_scores = np.abs((panel_pd[col] - mean) / std)
                if (z_scores > 3).any():
                    outliers_detected = True
                    # Clip to 99th percentile range
                    lower_bound = panel_pd[col].quantile(0.01)
                    upper_bound = panel_pd[col].quantile(0.99)
                    panel_pd[col] = panel_pd[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.debug(
                        "Clipped outliers in %s: [%.1f, %.1f] -> [%.1f, %.1f]",
                        col, original_range[0], original_range[1],
                        panel_pd[col].min(), panel_pd[col].max()
                    )

        if outliers_detected:
            logger.warning("Data quality filter: outliers clipped in %d ticker(s)",
                          sum(1 for col in tickers if
                              (np.abs((panel_original[col] - panel_original[col].mean()) /
                                     panel_original[col].std()) > 3).any()))

        # Adaptive HP lambda based on price volatility
        # Higher volatility -> higher lambda -> more smoothing -> smaller cycles
        panel_vol = float(panel_pd.std().mean())

        if panel_vol > 5.0:
            # High price volatility: smooth more aggressively
            hp_lambda = self.cfg.hp_lambda * 3.0  # 6.25 -> 18.75
            logger.info(
                "High price volatility (%.2fc) - increased HP lambda: %.1f -> %.1f",
                panel_vol, self.cfg.hp_lambda, hp_lambda
            )
        elif panel_vol > 3.0:
            # Moderate volatility: slight increase
            hp_lambda = self.cfg.hp_lambda * 1.5  # 6.25 -> 9.375
            logger.debug(
                "Moderate price volatility (%.2fc) - increased HP lambda: %.1f -> %.1f",
                panel_vol, self.cfg.hp_lambda, hp_lambda
            )
        else:
            # Normal volatility: use default
            hp_lambda = self.cfg.hp_lambda

        # Step 1: HP filter decomposition per ticker
        cycles = pd.DataFrame(index=panel_pd.index)
        trends = pd.DataFrame(index=panel_pd.index)
        for col in tickers:
            cycle, trend = hpfilter(panel_pd[col], lamb=hp_lambda)
            cycles[col] = cycle
            trends[col] = trend

        # Step 2: Check cycle volatility
        k_endog = cycles.shape[1]
        if k_endog < 2:
            raise RuntimeError(f"DFM needs >=2 tickers, got {k_endog}")

        cycle_vol = float(np.abs(cycles.to_numpy()).max()) or 1.0
        max_allowed = max(100.0, cycle_vol * 100.0)

        # Skip DFM if cycle volatility is too high (prevents divergence)
        HIGH_VOL_THRESHOLD = 5.0  # Normal < 1.0, high-vol ~8-10
        use_simple_model = cycle_vol > HIGH_VOL_THRESHOLD

        if use_simple_model:
            logger.warning(
                "HIGH CYCLE VOLATILITY (%.2f > %.2f) - using simple persistence model",
                cycle_vol, HIGH_VOL_THRESHOLD
            )

        # Step 3: Fit model (DFM or simple persistence)
        dfm_result = None

        if not use_simple_model:
            # Try DFM fitting
            k_factors = min(self.cfg.k_factors, k_endog - 1)

            # Adaptive factor_order: reduce for short time series
            n_obs = len(cycles)
            if n_obs < 25:
                factor_order = 1  # AR(1) for very short series
            elif n_obs < 40:
                factor_order = 2  # AR(2) for medium series
            else:
                factor_order = self.cfg.factor_order  # Full AR(3) for longer series

            # Try stationary fit first
            try:
                dfm = DynamicFactor(
                    cycles,
                    k_factors=k_factors,
                    factor_order=factor_order,
                    enforce_stationarity=True,
                )

                n_params = dfm.k_params
                start_params = np.full(n_params, 0.01)
                for i in range(k_endog):
                    start_params[n_params - k_endog + i] = max(cycles.iloc[:, i].var(), 1e-6)

                result = dfm.fit(
                    start_params=start_params, method="bfgs", disp=False, maxiter=2000
                )

                # Check forecast BEFORE accepting
                forecast = result.forecast(steps=1)
                fc_arr = forecast.to_numpy()

                if np.isfinite(fc_arr).all() and np.abs(fc_arr).max() <= max_allowed:
                    dfm_result = (result, forecast)
                    logger.debug("DFM stationary fit succeeded, AIC=%.2f", result.aic)
                else:
                    logger.warning(
                        "DFM stationary forecast diverged: max=%.2e > %.2e",
                        np.abs(fc_arr).max(), max_allowed
                    )

            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning("DFM stationary fit failed: %s", type(e).__name__)

            # If stationary failed, try non-stationary
            if dfm_result is None:
                try:
                    logger.warning("Retrying DFM without stationarity constraint...")
                    dfm_relaxed = DynamicFactor(
                        cycles,
                        k_factors=k_factors,
                        factor_order=factor_order,
                        enforce_stationarity=False,
                    )

                    result = dfm_relaxed.fit(
                        start_params=start_params, method="bfgs", disp=False, maxiter=2000
                    )

                    # Check forecast before accepting
                    forecast = result.forecast(steps=1)
                    fc_arr = forecast.to_numpy()

                    if np.isfinite(fc_arr).all() and np.abs(fc_arr).max() <= max_allowed:
                        dfm_result = (result, forecast)
                        logger.debug("DFM non-stationary fit succeeded, AIC=%.2f", result.aic)
                    else:
                        logger.warning(
                            "DFM non-stationary also diverged: max=%.2e > %.2e",
                            np.abs(fc_arr).max(), max_allowed
                        )

                except Exception as e:
                    logger.warning("DFM non-stationary fit failed: %s", type(e).__name__)

        # Step 4: Use DFM if successful, otherwise fall back to persistence
        if dfm_result is not None:
            result, forecast = dfm_result
            model_used = "DFM"
        else:
            # Simple persistence forecast with mean reversion
            logger.warning(
                "DFM unavailable (vol=%.2f) - using simple persistence forecast",
                cycle_vol
            )
            # Forecast = current cycle * damping (assumes 50% mean reversion per step)
            damping = 0.5
            forecast = cycles.iloc[[-1]] * damping
            fc_arr = forecast.to_numpy()
            model_used = "persistence"

            # Sanity check
            if not np.isfinite(fc_arr).all():
                raise RuntimeError(
                    "Persistence forecast is non-finite! Data quality issue."
                )

        # Step 5: Build per-ticker forecasts using empirical cycle std
        forecasts: dict[str, TickerForecast] = {}
        for col in tickers:
            res_std = float(cycles[col].std())
            if res_std < 1e-10:
                res_std = 1e-6
            cycle_arr = cycles[col].to_numpy()
            forecasts[col] = TickerForecast(
                ticker=col,
                current_cycle=float(cycles[col].iloc[-1]),
                forecast_cycle=float(forecast[col].iloc[0]),
                trend=float(trends[col].iloc[-1]),
                residual_std=res_std,
                recent_cycles=list(cycle_arr[-lookback:]),
            )

        self._last_result = forecasts

        # Log model results with applied fixes
        fixes_applied = []
        if outliers_detected:
            fixes_applied.append("data_filter")
        if hp_lambda != self.cfg.hp_lambda:
            fixes_applied.append(f"adaptive_lambda({hp_lambda:.1f})")
        if use_simple_model:
            fixes_applied.append("high_vol_skip")
        elif model_used == "persistence":
            fixes_applied.append("dfm_failed_fallback")

        fixes_str = f" [fixes: {', '.join(fixes_applied)}]" if fixes_applied else ""

        if model_used == "DFM":
            logger.info(
                "DFM fit complete: %d tickers, AIC=%.2f, cycle_vol=%.2f, price_vol=%.2f%s",
                len(tickers), result.aic, cycle_vol, panel_vol, fixes_str
            )
        else:
            logger.info(
                "Persistence model: %d tickers, cycle_vol=%.2f (threshold=%.2f), price_vol=%.2f%s",
                len(tickers), cycle_vol, HIGH_VOL_THRESHOLD, panel_vol, fixes_str
            )

        return forecasts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_engine(cfg: PipelineConfig) -> EWMAEngine | HPCycleDFMEngine:
    """Return the appropriate engine based on config."""
    if cfg.use_ewma:
        return EWMAEngine(cfg)
    return HPCycleDFMEngine(cfg)