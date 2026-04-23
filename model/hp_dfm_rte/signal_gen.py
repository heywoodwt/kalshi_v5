# pyright: reportArgumentType=false
"""Threshold-based signal logic with cooldown, price filters, and structured logging."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import cast

import numpy as np
from datetime import datetime, timezone
from enum import Enum

from .config import PipelineConfig
from .fees import round_trip_fee
from .model_engine import TickerForecast
from .orderbook import OrderbookSnapshot, extract_orderbook_snapshot

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    BUY_YES = "BUY_YES"    # z > 0: current_cycle < 0 (price depressed), expect reversion up
    BUY_NO = "BUY_NO"      # z < 0: current_cycle > 0 (price elevated), expect reversion down
    SELL_YES = "SELL_YES"   # exit a BUY_YES position
    SELL_NO = "SELL_NO"     # exit a BUY_NO position


@dataclass
class Signal:
    ticker: str
    direction: SignalDirection
    z_score: float
    current_cycle: float
    forecast_cycle: float
    residual_std: float
    trend: float
    is_exit: bool = False
    position_id: str = ""
    contracts: int = 1  # Position size scaled by z-score strength
    fair_value_prob: float | None = None
    edge_threshold_prob: float | None = None
    cycle_score: float | None = None
    quote_yes_bid_prob: float | None = None
    quote_yes_ask_prob: float | None = None
    edge_to_market_prob: float | None = None
    strong_edge: bool = False
    fillable: bool = False
    fill_price: float | None = None
    fill_size: int | None = None
    orderbook_age_s: float | None = None


# Regex to extract expiry hour from common Kalshi BTC ticker formats:
#   KXBTC-26MAR2123-B69050    -> hour 21 (from "2123")
#   KXBTCD-26MAR2123-T69399.99 -> hour 21
#   KXBTC15M-26MAR212315-15   -> hour 21
_EXPIRY_HOUR_RE = re.compile(
    r"KXBTC\w*-\d{2}[A-Z]{3}(\d{2})(\d{2})"
)


def _parse_expiry_hour_minute(ticker: str) -> tuple[int, int] | None:
    """Extract (hour, minute) from a Kalshi BTC ticker, or None if unparseable."""
    m = _EXPIRY_HOUR_RE.search(ticker)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


class SignalGenerator:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._last_signal_time: dict[str, float] = {}
        self._filter_stats: dict[str, int] = {
            "threshold_filtered": 0,
            "cooldown_filtered": 0,
            "price_band_filtered": 0,
            "expected_move_filtered": 0,
            "expiry_blackout_filtered": 0,
            "momentum_filtered": 0,
            "price_filtered": 0,
            "z_scaled_filtered": 0,
            "magnitude_filtered": 0,
            "expiry_filtered": 0,
            "vol_adaptive_filtered": 0,
            "forecast_swing_filtered": 0,
            "passed": 0,
        }
        self._last_run_funnel: dict[str, int] = {}

    def evaluate(
        self,
        forecasts: dict[str, TickerForecast],
        open_positions: list[dict] | None = None,
        prices: dict[str, float] | None = None,
        current_time: datetime | None = None,
        orderbooks: dict[str, OrderbookSnapshot | dict] | None = None,
    ) -> list[Signal]:
        """Evaluate forecasts and emit entry/exit signals.

        Args:
            forecasts: Per-ticker model forecasts.
            open_positions: Open positions for exit evaluation.
            prices: Current yes_price per ticker (cents). Used for price filters.
            orderbooks: Optional top-of-book snapshots per ticker.
            current_time: Current UTC time. Used for expiry blackout filter.
        """
        now = time.monotonic()
        signals: list[Signal] = []
        run_funnel = self._init_run_funnel()

        # Compute mean-reversion z-scores per ticker: how far is the current cycle
        # from zero, normalized by cycle std. Negated so that:
        #   z > 0  <=>  cycle < 0  (price below trend) -> BUY_YES (expect reversion up)
        #   z < 0  <=>  cycle > 0  (price above trend) -> BUY_NO  (expect reversion down)
        z_scores: dict[str, float] = {}
        for ticker, fc in forecasts.items():
            z_scores[ticker] = -fc.current_cycle / fc.residual_std

        # --- Exit signals (evaluated first so entries don't conflict) ---
        if self.cfg.exit_signals_enabled and open_positions:
            exit_signals = self._evaluate_exits(
                forecasts,
                z_scores,
                open_positions,
                prices,
                current_time,
                orderbooks,
            )
            signals.extend(exit_signals)

        # --- Entry signals ---
        for ticker, fc in forecasts.items():
            run_funnel["candidate_events"] += 1
            z_score = z_scores[ticker]

            if abs(z_score) <= self.cfg.signal_threshold:
                self._increment_filter_counters(run_funnel, "threshold_filtered")
                self._filter_stats["threshold_filtered"] += 1
                continue

            # Cooldown check
            last = self._last_signal_time.get(ticker, 0.0)
            if now - last < self.cfg.cooldown_s:
                self._increment_filter_counters(run_funnel, "cooldown_filtered")
                self._filter_stats["cooldown_filtered"] += 1
                continue

            # --- Filter 1: Price level filter ---
            yes_price = (prices or {}).get(ticker)
            if yes_price is not None:
                if not (self.cfg.price_filter_min <= yes_price <= self.cfg.price_filter_max):
                    self._increment_filter_counters(run_funnel, "price_band_filtered")
                    self._filter_stats["price_band_filtered"] += 1
                    self._filter_stats["price_filtered"] += 1
                    logger.debug(
                        "FILTERED (price) | %s | yes_price=%.1f outside [%.0f, %.0f]",
                        ticker, yes_price,
                        self.cfg.price_filter_min, self.cfg.price_filter_max,
                    )
                    continue

            # --- Filter 2: Z-score scaling by price distance from center ---
            if self.cfg.z_score_price_scaling and yes_price is not None:
                scaled_threshold = self._scaled_z_threshold(yes_price)
                if abs(z_score) <= scaled_threshold:
                    self._increment_filter_counters(run_funnel, "z_scaled_filtered")
                    self._filter_stats["z_scaled_filtered"] += 1
                    logger.debug(
                        "FILTERED (z-scale) | %s | |z|=%.3f < scaled_thresh=%.3f (price=%.1f)",
                        ticker, abs(z_score), scaled_threshold, yes_price,
                    )
                    continue

            # --- Filter 3: Magnitude filter (expected move > round-trip fees) ---
            if yes_price is not None:
                expected_move = fc.residual_std * abs(z_score) * 100  # convert to cents
                price_int = max(1, min(99, int(round(yes_price))))
                rt_fee = round_trip_fee(price_int, price_int, maker=self.cfg.assume_maker)
                min_profit = rt_fee * self.cfg.min_expected_move_multiple
                if expected_move < min_profit:
                    self._increment_filter_counters(run_funnel, "expected_move_filtered")
                    self._filter_stats["expected_move_filtered"] += 1
                    self._filter_stats["magnitude_filtered"] += 1
                    logger.debug(
                        "FILTERED (magnitude) | %s | exp_move=%.3f < %.1f (%.1f * rt_fee=%d)",
                        ticker, expected_move, min_profit,
                        self.cfg.min_expected_move_multiple, rt_fee,
                    )
                    continue

                # Edge filter - expected profit after fees must exceed min_edge_cents
                expected_profit = expected_move - rt_fee
                if expected_profit < self.cfg.min_edge_cents:
                    self._increment_filter_counters(run_funnel, "expected_move_filtered")
                    self._filter_stats["expected_move_filtered"] += 1
                    self._filter_stats["magnitude_filtered"] += 1
                    logger.debug(
                        "FILTERED (edge) | %s | exp_profit=%.3f < min_edge=%.1fc",
                        ticker, expected_profit, self.cfg.min_edge_cents,
                    )
                    continue

            # --- Filter 4: Expiry blackout ---
            if current_time is not None and self.cfg.expiry_blackout_minutes > 0:
                if self._is_near_expiry(ticker, current_time):
                    self._increment_filter_counters(run_funnel, "expiry_blackout_filtered")
                    self._filter_stats["expiry_blackout_filtered"] += 1
                    self._filter_stats["expiry_filtered"] += 1
                    logger.debug(
                        "FILTERED (expiry) | %s | within %d min of expiry",
                        ticker, self.cfg.expiry_blackout_minutes,
                    )
                    continue

            # --- Filter 5: Momentum filter -- don't catch a falling knife ---
            if self.cfg.momentum_filter_enabled and len(fc.recent_cycles) >= 3:
                recent = fc.recent_cycles[-3:]
                still_falling = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
                still_rising = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
                if z_score > 0 and still_falling:
                    self._increment_filter_counters(run_funnel, "momentum_filtered")
                    self._filter_stats["momentum_filtered"] += 1
                    logger.debug("FILTERED (momentum) | %s | cycle still falling z=%.3f", ticker, z_score)
                    continue
                if z_score < 0 and still_rising:
                    self._increment_filter_counters(run_funnel, "momentum_filtered")
                    self._filter_stats["momentum_filtered"] += 1
                    logger.debug("FILTERED (momentum) | %s | cycle still rising z=%.3f", ticker, z_score)
                    continue

            # --- Filter 6: Volatility-adaptive threshold ---
            if self.cfg.vol_adaptive_threshold_enabled and len(fc.recent_cycles) >= self.cfg.vol_short_period:
                short_vol = float(np.std(fc.recent_cycles[-self.cfg.vol_short_period:]))
                n_medium = min(len(fc.recent_cycles), self.cfg.vol_medium_period)
                medium_vol = float(np.std(fc.recent_cycles[-n_medium:])) if n_medium >= 2 else short_vol
                if medium_vol > 1e-6:
                    vol_ratio = short_vol / medium_vol
                    # Scale threshold between 0.75x (low vol) and 2.0x (high vol)
                    adaptive_threshold = self.cfg.signal_threshold * max(0.75, min(2.0, vol_ratio))
                    if abs(z_score) <= adaptive_threshold:
                        self._increment_filter_counters(run_funnel, "vol_adaptive_filtered")
                        self._filter_stats["vol_adaptive_filtered"] += 1
                        logger.debug(
                            "FILTERED (vol-adaptive) | %s | |z|=%.3f < adaptive=%.3f (vol_ratio=%.2f)",
                            ticker, abs(z_score), adaptive_threshold, vol_ratio,
                        )
                        continue

            # --- Filter 7: Forecast swing filter ---
            # Reject signals where the forecast predicts overshoot or divergence
            current_cycle = fc.current_cycle
            forecast_cycle = fc.forecast_cycle

            # Check if forecast crosses zero (opposite signs)
            crosses_zero = (current_cycle * forecast_cycle) < 0

            # Check if forecast is diverging (moving away from mean = zero)
            if current_cycle < 0:
                is_diverging = forecast_cycle < current_cycle
            else:
                is_diverging = forecast_cycle > current_cycle

            if crosses_zero or is_diverging:
                self._increment_filter_counters(run_funnel, "forecast_swing_filtered")
                self._filter_stats["forecast_swing_filtered"] += 1
                reason = "OVERSHOOT" if crosses_zero else "DIVERGING"
                logger.debug(
                    "FILTERED (forecast-%s) | %s | cycle=%.4f->%.4f | z=%.3f",
                    reason, ticker, current_cycle, forecast_cycle, z_score,
                )
                continue

            self._filter_stats["passed"] += 1
            run_funnel["accepted_trades"] += 1

            direction = (
                SignalDirection.BUY_YES if z_score > 0 else SignalDirection.BUY_NO
            )

            fair_value_prob = None
            edge_threshold_prob = None
            cycle_score = None
            quote_yes_bid_prob = None
            quote_yes_ask_prob = None
            edge_to_market_prob = None
            strong_edge = False

            if self.cfg.mm_enabled and yes_price is not None:
                fair_value_prob, edge_threshold_prob, cycle_score, quote_yes_bid_prob, quote_yes_ask_prob, edge_to_market_prob = self._build_quote_levels(fc, yes_price)
                strong_edge = edge_to_market_prob > (self.cfg.mm_extreme_edge_multiple * edge_threshold_prob)

            contracts = self._compute_contracts(z_score, edge_to_market_prob, edge_threshold_prob)

            signal = Signal(
                ticker=ticker,
                direction=direction,
                z_score=z_score,
                current_cycle=fc.current_cycle,
                forecast_cycle=fc.forecast_cycle,
                residual_std=fc.residual_std,
                trend=fc.trend,
                contracts=contracts,
                fair_value_prob=fair_value_prob,
                edge_threshold_prob=edge_threshold_prob,
                cycle_score=cycle_score,
                quote_yes_bid_prob=quote_yes_bid_prob,
                quote_yes_ask_prob=quote_yes_ask_prob,
                edge_to_market_prob=edge_to_market_prob,
                strong_edge=strong_edge,
            )
            signals.append(signal)
            self._last_signal_time[ticker] = now

            logger.warning(
                "SIGNAL | %s | %s | z=%.3f | cycle=%.4f->%.4f | std=%.4f | trend=%.4f | contracts=%d",
                signal.direction.value,
                signal.ticker,
                signal.z_score,
                signal.current_cycle,
                signal.forecast_cycle,
                signal.residual_std,
                signal.trend,
                signal.contracts,
            )

        self._emit_funnel_report(run_funnel)
        return signals

    def _init_run_funnel(self) -> dict[str, int]:
        return {
            "candidate_events": 0,
            "threshold_filtered": 0,
            "cooldown_filtered": 0,
            "price_band_filtered": 0,
            "z_scaled_filtered": 0,
            "expected_move_filtered": 0,
            "expiry_blackout_filtered": 0,
            "momentum_filtered": 0,
            "vol_adaptive_filtered": 0,
            "forecast_swing_filtered": 0,
            "accepted_trades": 0,
        }

    def _increment_filter_counters(self, run_funnel: dict[str, int], key: str) -> None:
        run_funnel[key] += 1

    def _emit_funnel_report(self, run_funnel: dict[str, int]) -> None:
        self._last_run_funnel = dict(run_funnel)
        candidate_events = run_funnel.get("candidate_events", 0)
        if candidate_events <= 0:
            return

        ordered_keys = [
            "threshold_filtered",
            "cooldown_filtered",
            "price_band_filtered",
            "z_scaled_filtered",
            "expected_move_filtered",
            "expiry_blackout_filtered",
            "momentum_filtered",
            "vol_adaptive_filtered",
            "forecast_swing_filtered",
        ]
        parts = []
        for key in ordered_keys:
            dropped = run_funnel.get(key, 0)
            pct = (dropped / candidate_events) * 100.0
            parts.append(f"{key}={dropped} ({pct:.1f}%)")
        accepted = run_funnel.get("accepted_trades", 0)
        accepted_pct = (accepted / candidate_events) * 100.0
        logger.info(
            "FUNNEL REPORT | candidates=%d | %s | accepted=%d (%.1f%%)",
            candidate_events,
            " | ".join(parts),
            accepted,
            accepted_pct,
        )

    def _scaled_z_threshold(self, yes_price: float) -> float:
        """Require stronger z-scores for prices far from 50 cents.

        At 50c (center): use base threshold.
        At 20c or 80c (edges): require 1.5x the base threshold.
        """
        distance_from_center = abs(yes_price - 50.0) / 30.0  # 0 at 50c, 1 at 20c/80c
        distance_from_center = min(distance_from_center, 1.0)
        scale = 1.0 + 0.5 * distance_from_center  # 1.0x at center, 1.5x at edges
        return self.cfg.signal_threshold * scale

    def _clip_prob(self, value: float) -> float:
        return max(0.01, min(0.99, value))

    def _compute_fair_value_prob(self, fc: TickerForecast) -> float:
        hp_trend = self._clip_prob(fc.trend / 100.0)
        tft_trend = getattr(fc, "tft_trend", hp_trend)
        tft_trend = self._clip_prob(float(tft_trend))
        w = max(0.0, min(1.0, self.cfg.mm_tft_weight))
        return self._clip_prob((w * tft_trend) + ((1.0 - w) * hp_trend))

    def _compute_cycle_score(self, fc: TickerForecast) -> float:
        hp_cycle = float(fc.current_cycle)
        tft_cycle = float(getattr(fc, "tft_cycle", hp_cycle))
        w = max(0.0, min(1.0, self.cfg.mm_cycle_tft_weight))
        blended = (w * tft_cycle) + ((1.0 - w) * hp_cycle)
        denom = max(abs(fc.residual_std) * max(self.cfg.mm_cycle_norm_scale, 1e-6), 1e-6)
        return float(np.tanh(blended / denom))

    def _compute_edge_threshold_prob(self, fair_value_prob: float) -> float:
        fair_cents = int(round(self._clip_prob(fair_value_prob) * 100.0))
        rt_fee_cents = round_trip_fee(
            fair_cents,
            fair_cents,
            contracts=max(1, self.cfg.mm_base_contracts),
            maker=self.cfg.assume_maker,
        )
        fee_prob = rt_fee_cents / 100.0
        return max(self.cfg.mm_edge_fee_multiple * fee_prob, self.cfg.mm_min_edge_prob)

    def _build_quote_levels(self, fc: TickerForecast, market_yes_cents: float) -> tuple[float, float, float, float, float, float]:
        fair_value_prob = self._compute_fair_value_prob(fc)
        cycle_score = self._compute_cycle_score(fc)
        edge_threshold_prob = self._compute_edge_threshold_prob(fair_value_prob)

        bid_prob = fair_value_prob - edge_threshold_prob - (self.cfg.mm_cycle_shift_k * cycle_score)
        ask_prob = fair_value_prob + edge_threshold_prob - (self.cfg.mm_cycle_shift_k * cycle_score)
        bid_prob = self._clip_prob(bid_prob)
        ask_prob = self._clip_prob(max(ask_prob, bid_prob + 0.001))

        market_prob = self._clip_prob(market_yes_cents / 100.0)
        edge_to_market_prob = abs(fair_value_prob - market_prob)
        return fair_value_prob, edge_threshold_prob, cycle_score, bid_prob, ask_prob, edge_to_market_prob

    def _compute_contracts(
        self,
        z_score: float,
        edge_to_market_prob: float | None,
        edge_threshold_prob: float | None,
    ) -> int:
        if self.cfg.mm_enabled and edge_to_market_prob is not None and edge_threshold_prob is not None:
            excess_edge = max(0.0, edge_to_market_prob - edge_threshold_prob)
            edge_step = max(self.cfg.mm_size_edge_step_prob, 1e-6)
            edge_units = excess_edge / edge_step
            sized = int(np.ceil(self.cfg.mm_base_contracts * max(1.0, edge_units)))
            return min(self.cfg.max_contracts, max(1, sized))

        # Fallback sizing if market-making edge inputs are unavailable
        z_multiple = abs(z_score) / self.cfg.signal_threshold
        return min(self.cfg.max_contracts, max(1, int(z_multiple)))

    def _is_near_expiry(self, ticker: str, current_time: datetime) -> bool:
        """Check if the current time is within expiry_blackout_minutes of this contract's expiry."""
        parsed = _parse_expiry_hour_minute(ticker)
        if parsed is None:
            return False
        exp_hour, exp_minute = parsed

        expiry = current_time.replace(
            hour=exp_hour, minute=exp_minute, second=0, microsecond=0,
        )
        if expiry < current_time:
            return False

        minutes_to_expiry = (expiry - current_time).total_seconds() / 60.0
        return minutes_to_expiry <= self.cfg.expiry_blackout_minutes

    def get_filter_stats(self) -> dict[str, int]:
        """Return cumulative filter statistics."""
        stats = dict(self._filter_stats)
        stats["last_run_candidates"] = self._last_run_funnel.get("candidate_events", 0)
        stats["last_run_accepted"] = self._last_run_funnel.get("accepted_trades", 0)
        return stats

    def _coerce_orderbook_snapshot(
        self,
        orderbook: OrderbookSnapshot | dict | None,
        current_time: datetime | None = None,
    ) -> OrderbookSnapshot | None:
        """Normalize raw orderbook payloads into a compact snapshot."""
        if orderbook is None:
            return None
        if isinstance(orderbook, OrderbookSnapshot):
            return orderbook
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        return extract_orderbook_snapshot(orderbook, updated_at=current_time)

    def _book_price_for_position(
        self,
        snapshot: OrderbookSnapshot,
        direction: str,
    ) -> tuple[float | None, int]:
        """Return the top executable price/size for the open position's side."""
        if direction == "BUY_YES":
            return snapshot.yes_price, snapshot.yes_size
        return snapshot.no_price, snapshot.no_size

    def _get_fillable_book_price(
        self,
        ticker: str,
        direction: str,
        contracts: int,
        snapshot: OrderbookSnapshot | None,
        current_time: datetime,
    ) -> tuple[float | None, int | None, float | None]:
        """Return a fresh top-of-book price if the position can likely be filled."""
        if snapshot is None:
            return None, None, None

        age_s = snapshot.age_s(current_time)
        if age_s is None or age_s > self.cfg.orderbook_max_age_s:
            logger.debug(
                "SKIP EXIT | ticker=%s | stale orderbook age=%s",
                ticker,
                "unknown" if age_s is None else f"{age_s:.1f}s",
            )
            return None, None, age_s

        price, size = self._book_price_for_position(snapshot, direction)
        if price is None:
            logger.debug("SKIP EXIT | ticker=%s | orderbook missing side price", ticker)
            return None, None, age_s

        if size < max(contracts, self.cfg.orderbook_min_size):
            logger.debug(
                "SKIP EXIT | ticker=%s | depth=%d below required=%d",
                ticker,
                size,
                max(contracts, self.cfg.orderbook_min_size),
            )
            return None, size, age_s

        # Use a tiny slippage buffer so we only mark the trade fillable when the
        # top of book still gives us a realistic executable price.
        fill_price = max(0.0, price - self.cfg.orderbook_slippage_cents)
        return fill_price, size, age_s

    def _evaluate_exits(
        self,
        forecasts: dict[str, TickerForecast],
        z_scores: dict[str, float],
        open_positions: list[dict],
        prices: dict[str, float] | None = None,
        current_time: datetime | None = None,
        orderbooks: dict[str, OrderbookSnapshot | dict] | None = None,
    ) -> list[Signal]:
        """Exit logic: profit target, stop loss, trailing stop, z-score, time-based, expiry.

        Exit priority:
        1. Profit target (>=5c profit)
        2. Stop loss (>=-10c loss)
        3. Trailing stop (lock in 70% of peak profit)
        4. Z-score reversal (mean reversion complete)
        5. Time-based (after 20 minutes)
        6. Expiry approach (2 minutes before expiry)
        """
        exit_signals: list[Signal] = []
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        effective_time = current_time

        for pos in open_positions:
            ticker = pos["ticker"]
            fc = forecasts.get(ticker)
            if fc is None:
                continue

            z_score = z_scores[ticker]
            direction = pos["direction"]
            entry_price = float(pos.get("entry_price", 0) or 0)
            contracts = int(pos.get("contracts", 1))

            logger.debug(
                "EXIT EVAL | ticker=%s | direction=%s | entry_price=%.1f | pos_id=%s",
                ticker, direction, entry_price, pos.get("position_id", "unknown")[:8]
            )

            # Prefer the orderbook top-of-book for fillable exits.
            current_side_price: float | None = None
            current_side_size: int | None = None
            orderbook_age_s: float | None = None
            snapshot = self._coerce_orderbook_snapshot((orderbooks or {}).get(ticker), current_time=current_time)
            if self.cfg.orderbook_enabled and snapshot is not None:
                current_side_price, current_side_size, orderbook_age_s = self._get_fillable_book_price(
                    ticker,
                    direction,
                    contracts,
                    snapshot,
                    effective_time,
                )
                if current_side_price is None:
                    continue

            # Fall back to ticker prices when we do not have a fresh, deep book.
            used_fallback = current_side_price is None
            if current_side_price is None:
                current_side_price = (prices or {}).get(ticker)
                if current_side_price is None:
                    # No live price, so approximate the mark using the entry price.
                    current_side_price = entry_price
            price_for_pnl = cast(float, float(current_side_price))

            # Calculate unrealized P&L (cents, before fees)
            price_change = price_for_pnl - entry_price

            # Estimate fees (2c round-trip for typical prices)
            estimated_fees = 2.0
            unrealized_pnl = price_change - estimated_fees

            # Don't emit an exit unless the book or price moved enough to justify fees.
            # Exception: allow exit if contract is near/past expiry or price is missing.
            minutes_to_expiry = self._minutes_to_expiry(ticker, effective_time)
            is_near_expiry = False
            if minutes_to_expiry is not None:
                expiry_minutes = float(minutes_to_expiry)
                is_near_expiry = expiry_minutes <= float(self.cfg.expiry_exit_minutes)
            is_missing_price = used_fallback

            if abs(price_change) < self.cfg.min_price_movement_cents and not is_near_expiry and not is_missing_price:
                logger.debug(
                    "SKIP EXIT | ticker=%s | price_change=%.2fc < min=%.2fc",
                    ticker, price_change, self.cfg.min_price_movement_cents
                )
                continue
            elif is_missing_price:
                logger.warning(
                    "FORCE EXIT (missing price) | ticker=%s | likely expired/delisted",
                    ticker
                )

            # Update peak profit tracking
            peak_profit = pos.get("peak_profit", unrealized_pnl)
            peak_profit = max(peak_profit, unrealized_pnl)
            pos["peak_profit"] = peak_profit

            # Parse opened_at timestamp
            opened_at = pos.get("opened_at")
            if opened_at:
                if isinstance(opened_at, str):
                    opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                elif not isinstance(opened_at, datetime):
                    opened_at = datetime.fromtimestamp(opened_at, tz=timezone.utc)

            holding_minutes = (effective_time - opened_at).total_seconds() / 60.0 if opened_at else 0

            exit_reason = None
            exit_dir = SignalDirection.SELL_YES if direction == "BUY_YES" else SignalDirection.SELL_NO

            # EXIT 0: Missing Price (likely expired/delisted contract)
            if is_missing_price:
                exit_reason = "MISSING_PRICE"
                logger.warning(
                    "EXIT SIGNAL (EXPIRED) | %s | %s | profit=%.1fc",
                    exit_dir.value, ticker, float(unrealized_pnl),
                )

            # EXIT 1: Profit Target
            if unrealized_pnl >= self.cfg.profit_target_cents:
                exit_reason = "PROFIT_TARGET"
                logger.warning(
                    "EXIT SIGNAL (PROFIT) | %s | %s | profit=%.1fc | held=%.1fmin",
                    exit_dir.value, ticker, float(unrealized_pnl), float(holding_minutes),
                )

            # EXIT 2: Stop Loss
            elif unrealized_pnl <= -self.cfg.stop_loss_cents:
                exit_reason = "STOP_LOSS"
                logger.warning(
                    "EXIT SIGNAL (STOP) | %s | %s | loss=%.1fc | held=%.1fmin",
                    exit_dir.value, ticker, float(unrealized_pnl), float(holding_minutes),
                )

            # EXIT 3: Trailing Stop
            elif peak_profit >= self.cfg.trailing_stop_cents:
                trailing_threshold = peak_profit * self.cfg.trailing_stop_pct
                if unrealized_pnl < trailing_threshold:
                    exit_reason = "TRAILING_STOP"
                    logger.warning(
                        "EXIT SIGNAL (TRAIL) | %s | %s | profit=%.1fc peak=%.1fc thresh=%.1fc",
                        exit_dir.value,
                        ticker,
                        float(unrealized_pnl),
                        float(peak_profit),
                        float(trailing_threshold),
                    )

            # EXIT 4: Z-Score Reversal (mean reversion complete)
            elif (direction == "BUY_YES" and z_score < self.cfg.exit_z_threshold) or \
                 (direction == "BUY_NO" and z_score > -self.cfg.exit_z_threshold):
                exit_reason = "Z_SCORE_REVERSAL"
                logger.warning(
                    "EXIT SIGNAL (Z-SCORE) | %s | %s | z=%.3f | profit=%.1fc | held=%.1fmin",
                    exit_dir.value, ticker, float(z_score), float(unrealized_pnl), float(holding_minutes),
                )

            # EXIT 5: Time-Based
            elif holding_minutes >= self.cfg.holding_period_minutes:
                exit_reason = "TIME_EXPIRED"
                logger.warning(
                    "EXIT SIGNAL (TIME) | %s | %s | profit=%.1fc | held=%.1fmin",
                    exit_dir.value, ticker, float(unrealized_pnl), float(holding_minutes),
                )

            # EXIT 6: Expiry Approach
            elif self.cfg.expiry_exit_minutes > 0:
                minutes_to_expiry = self._minutes_to_expiry(ticker, effective_time)
                if minutes_to_expiry is not None:
                    expiry_minutes = float(minutes_to_expiry)
                    if expiry_minutes <= float(self.cfg.expiry_exit_minutes):
                        exit_reason = "EXPIRY_APPROACH"
                        logger.warning(
                            "EXIT SIGNAL (EXPIRY) | %s | %s | profit=%.1fc | expiry_in=%.1fmin",
                            exit_dir.value, ticker, float(unrealized_pnl), expiry_minutes,
                        )

            # Create exit signal if any condition met
            if exit_reason:
                fill_price_value: float = float(price_for_pnl)
                book_age_value: float | None = float(orderbook_age_s) if orderbook_age_s is not None else None
                signal = Signal(
                    ticker=ticker,
                    direction=exit_dir,
                    z_score=z_score,
                    current_cycle=fc.current_cycle,
                    forecast_cycle=fc.forecast_cycle,
                    residual_std=fc.residual_std,
                    trend=fc.trend,
                    is_exit=True,
                    position_id=pos["position_id"],
                    fillable=(not is_missing_price),
                    fill_price=fill_price_value,  # type: ignore[arg-type]
                    fill_size=int(current_side_size) if current_side_size is not None else None,
                    orderbook_age_s=book_age_value,  # type: ignore[arg-type]
                )
                exit_signals.append(signal)

        return exit_signals

    def _minutes_to_expiry(self, ticker: str, current_time: datetime) -> float | None:
        """Calculate minutes until contract expiry."""
        parsed = _parse_expiry_hour_minute(ticker)
        if parsed is None:
            return None

        exp_hour, exp_minute = parsed
        expiry = current_time.replace(
            hour=exp_hour, minute=exp_minute, second=0, microsecond=0,
        )

        if expiry < current_time:
            return None

        return (expiry - current_time).total_seconds() / 60.0
