"""Real-time paper trading on live Kalshi BTC markets via WebSocket.

Connects to Kalshi's authenticated WebSocket feed, accumulates per-ticker
price history, periodically runs HP-DFM-RTE model fitting and signal
generation, and simulates trades with PnL tracking.

No real orders are submitted. All trades are logged to stdout.

Usage:
    python live_paper_trade.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl

from authentication_to_kalshi.auth import load_private_key, make_ws_headers
from model.hp_dfm_rte.config import PipelineConfig
from model.hp_dfm_rte.model_engine import make_engine
from model.hp_dfm_rte.signal_gen import SignalGenerator, Signal
from model.hp_dfm_rte.fees import round_trip_fee, net_pnl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("live_paper")

# ── Config ───────────────────────────────────────────────────────────────────

# Minimum ticks per ticker before model fitting is attempted
MIN_OBS = 10

# How often (seconds) to run model fit + signal evaluation
EVAL_INTERVAL_S = 60  # Was 30; slower cadence reduces z-score noise between evals

# Maximum ticks to retain per ticker (rolling window)
MAX_BUFFER = 500

# Market prefix to filter for
MARKET_PREFIX = "KXBTC"

# Ticker type classification: longest prefixes first so KXBTCD- matches before KXBTC-.
# DFM assumes a common factor across the panel; mixing 15-min, hourly, daily, and
# monthly contracts violates that assumption → LinAlgError → persistence fallback.
_TICKER_PREFIX_TABLE: list[tuple[str, str]] = [
    ("KXBTC15M-", "15min"),
    ("KXBTCD-",   "daily"),
    ("KXBTCM-",   "monthly"),
    ("KXBTC-",    "hourly"),
]


def _classify_ticker(ticker: str) -> str | None:
    """Map a Kalshi BTC ticker to its contract type, or None if unknown."""
    for prefix, label in _TICKER_PREFIX_TABLE:
        if ticker.startswith(prefix):
            return label
    return None


# ── Position Tracking ────────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    """Tracks a single simulated position."""
    ticker: str
    direction: str          # "BUY_YES" or "BUY_NO"
    entry_price: float      # cents
    contracts: int
    opened_at: datetime
    position_id: str
    peak_profit: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade for PnL ledger."""
    ticker: str
    direction: str
    entry_price: float
    exit_price: float
    contracts: int
    gross_pnl: float
    fees: float
    net_pnl: float
    held_seconds: float
    exit_reason: str


# ── Live Paper Trader ────────────────────────────────────────────────────────

class LivePaperTrader:
    """Streams live Kalshi data, runs HP-DFM-RTE, paper-trades signals."""

    def __init__(self) -> None:
        self.cfg = PipelineConfig.from_env()
        # Override cooldown for live (prevent rapid re-entry on same ticker)
        self.cfg.cooldown_s = 60
        self.engine = make_engine(self.cfg)
        self.signal_gen = SignalGenerator(self.cfg)

        # Per-ticker price buffer: ticker -> deque of (timestamp, yes_price_cents)
        self._buffers: Dict[str, deque] = {}
        # Latest price per ticker (cents)
        self._prices: Dict[str, float] = {}
        # Open paper positions
        self._positions: Dict[str, PaperPosition] = {}
        # Completed trade ledger
        self._trades: list[TradeRecord] = []
        # Discovered tickers
        self._known_tickers: set[str] = set()
        # Timing
        self._last_eval = 0.0
        self._start_time = time.monotonic()
        self._tick_count = 0
        # Signal to stop
        self._running = True
        # Position id counter
        self._pos_counter = 0

    # ── WebSocket Connection ─────────────────────────────────────────────

    async def run(self) -> None:
        """Connect to Kalshi WebSocket and process messages."""
        import websockets

        api_key = self.cfg.kalshi_api_key_id
        key_path = self.cfg.kalshi_private_key_path
        if not api_key or not key_path:
            log.error("Missing PROD_API_KEY or PROD_KEY_PATH in .env")
            return

        private_key = load_private_key(key_path)
        ws_url = self.cfg.ws_url

        log.info("=" * 70)
        log.info("  LIVE PAPER TRADING — HP-DFM-RTE on Kalshi BTC Markets")
        log.info("  NO REAL ORDERS WILL BE SUBMITTED")
        log.info("=" * 70)
        self._print_config()

        delay = 1
        while self._running:
            try:
                headers = make_ws_headers(api_key, private_key)
                async with websockets.connect(ws_url, additional_headers=headers) as ws:
                    log.info("Connected to %s", ws_url)
                    delay = 1

                    # Subscribe to global ticker channel
                    await ws.send(json.dumps({
                        "id": 1,
                        "cmd": "subscribe",
                        "params": {"channels": ["ticker"]},
                    }))
                    log.info("Subscribed to global ticker channel")

                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        msg_type = msg.get("type")
                        if msg_type == "ticker":
                            self._on_ticker(msg)
                        elif msg_type == "trade":
                            self._on_trade(msg)

            except Exception as e:
                log.warning("Connection lost: %s. Reconnecting in %ds...", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    # ── Message Handlers ─────────────────────────────────────────────────

    def _on_ticker(self, msg: dict) -> None:
        """Process ticker update: buffer price, maybe run eval."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not ticker.startswith(MARKET_PREFIX):
            return

        # Extract price in cents from dollar-denominated fields
        # API sends yes_ask_dollars / yes_bid_dollars / price_dollars as strings like "0.0300"
        raw = (
            data.get("yes_ask_dollars")
            or data.get("yes_bid_dollars")
            or data.get("price_dollars")
            # Fallback to legacy field names
            or data.get("yes_ask")
            or data.get("yes_bid")
            or data.get("last_price")
        )
        if raw is None:
            return
        try:
            price_dollars = float(raw)
        except (TypeError, ValueError):
            return

        # Convert dollars to cents (0.03 -> 3c)
        price = price_dollars * 100.0 if price_dollars <= 1.0 else price_dollars

        if price < 1.0 or price > 99.0:
            return

        # Discover new tickers
        if ticker not in self._known_tickers:
            self._known_tickers.add(ticker)
            log.info("Discovered market: %s (price=%.1fc)", ticker, price)

        # Buffer the observation
        self._buffer_price(ticker, price)
        self._prices[ticker] = price
        self._tick_count += 1

        # Periodic evaluation
        now = time.monotonic()
        if now - self._last_eval >= EVAL_INTERVAL_S:
            self._run_evaluation()
            self._last_eval = now

    def _on_trade(self, msg: dict) -> None:
        """Process trade message: update price."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not ticker.startswith(MARKET_PREFIX):
            return

        raw = data.get("yes_price_dollars") or data.get("yes_price")
        if raw is None:
            return
        try:
            price_val = float(raw)
        except (TypeError, ValueError):
            return

        # Convert dollars to cents
        price = price_val * 100.0 if price_val <= 1.0 else price_val

        if 1.0 <= price <= 99.0:
            self._buffer_price(ticker, price)
            self._prices[ticker] = price

    # ── Price Buffering ──────────────────────────────────────────────────

    def _buffer_price(self, ticker: str, price_cents: float) -> None:
        """Add a price to the per-ticker rolling buffer."""
        if ticker not in self._buffers:
            self._buffers[ticker] = deque(maxlen=MAX_BUFFER)
        self._buffers[ticker].append((datetime.now(timezone.utc), price_cents))

    # ── Model Evaluation ─────────────────────────────────────────────────

    def _run_evaluation(self) -> None:
        """Build price panel, fit HP-DFM-RTE, generate signals, manage positions."""
        # Log buffer sizes for diagnostics
        buf_sizes = {t: len(b) for t, b in self._buffers.items()}
        top_5 = sorted(buf_sizes.items(), key=lambda x: -x[1])[:5]
        log.info("EVAL | total_tickers=%d total_ticks=%d | top buffers: %s",
                 len(self._buffers), self._tick_count,
                 ", ".join(f"{t}={n}" for t, n in top_5))

        # Need at least 2 tickers with enough observations
        eligible = {
            t: buf for t, buf in self._buffers.items()
            if len(buf) >= MIN_OBS
        }
        if len(eligible) < 2:
            log.info("Waiting for data: %d tickers with %d+ obs (need 2+)",
                     len(eligible), MIN_OBS)
            return

        # Build wide-format price panel from buffers
        panel = self._build_panel(eligible)
        if panel is None or panel.height < MIN_OBS:
            log.info("Panel too small: %s rows (need %d+)",
                     panel.height if panel is not None else 0, MIN_OBS)
            return

        log.info("MODEL FIT | panel=%d rows x %d tickers", panel.height, len(panel.columns) - 1)

        # Fit model and get forecasts
        try:
            forecasts = self.engine.fit_and_forecast(panel)
        except Exception as e:
            log.warning("Model fit failed: %s", e)
            return

        if not forecasts:
            return

        # Prepare open positions for exit evaluation
        open_pos_list = [
            {
                "ticker": p.ticker,
                "direction": p.direction,
                "entry_price": p.entry_price,
                "contracts": p.contracts,
                "opened_at": p.opened_at,
                "position_id": p.position_id,
                "peak_profit": p.peak_profit,
            }
            for p in self._positions.values()
        ]

        # Generate signals
        now_utc = datetime.now(timezone.utc)
        signals = self.signal_gen.evaluate(
            forecasts,
            open_positions=open_pos_list if open_pos_list else None,
            prices=self._prices,
            current_time=now_utc,
        )

        # Process signals
        for sig in signals:
            if sig.is_exit:
                self._process_exit(sig)
            else:
                self._process_entry(sig)

        # Print status
        self._print_status()

    def _build_panel(self, eligible: dict) -> pl.DataFrame | None:
        """Build a wide-format price panel from buffered data.

        Selects the top N most-active tickers (by buffer size), aligns to
        a common time grid (1-minute buckets), forward-fills, and drops only
        rows where the selected tickers have nulls.
        """
        # Group eligible tickers by contract type so DFM fits a homogeneous panel.
        # Mixing 15-min, hourly, daily contracts violates the common-factor assumption.
        MAX_PANEL_TICKERS = 10
        groups: dict[str, list[tuple[str, deque]]] = {}
        for t, buf in eligible.items():
            ttype = _classify_ticker(t)
            if ttype is None:
                continue
            groups.setdefault(ttype, []).append((t, buf))

        if not groups:
            log.info("No classifiable tickers in eligible set")
            return None

        # Pick the largest same-type group
        group_summary = {k: len(v) for k, v in groups.items()}
        best_type = max(groups, key=lambda k: len(groups[k]))
        log.info("Panel groups: %s | selected type=%s (%d tickers)",
                 group_summary, best_type, len(groups[best_type]))

        # Within that group, take top N by observation count
        sorted_tickers = sorted(groups[best_type], key=lambda x: -len(x[1]))
        selected = dict(sorted_tickers[:MAX_PANEL_TICKERS])

        # Need at least 2 tickers
        if len(selected) < 2:
            return None

        # Use 10-second buckets for finer time resolution.
        # With 1-minute buckets we only get ~5 rows in 5 minutes, too few
        # after drop_nulls removes leading rows for late-arriving tickers.
        BUCKET_SECONDS = 10

        rows = []
        for ticker, buf in selected.items():
            for ts, price in buf:
                # Truncate to 10-second bucket
                bucket_s = ts.second - (ts.second % BUCKET_SECONDS)
                ts_bucket = ts.replace(second=bucket_s, microsecond=0)
                rows.append({
                    "time_bucket": ts_bucket,
                    "ticker": ticker,
                    "yes_price": price,
                })

        if not rows:
            return None

        df = pl.DataFrame(rows)

        # Take last price per (time_bucket, ticker) to handle duplicates
        df = df.group_by(["time_bucket", "ticker"]).agg(
            pl.col("yes_price").last()
        )

        # Pivot to wide format
        wide = df.pivot(on="ticker", index="time_bucket", values="yes_price").sort("time_bucket")

        # Forward-fill and drop rows where any selected ticker is still null
        for col in wide.columns:
            if col != "time_bucket":
                wide = wide.with_columns(pl.col(col).forward_fill())
        wide = wide.drop_nulls()

        return wide if wide.height >= MIN_OBS else None

    # ── Position Management ──────────────────────────────────────────────

    def _process_entry(self, sig: Signal) -> None:
        """Open a new paper position from an entry signal."""
        # Skip if we already have a position on this ticker
        if sig.ticker in self._positions:
            return

        price = self._prices.get(sig.ticker)
        if price is None:
            return

        self._pos_counter += 1
        pos_id = f"paper-{self._pos_counter:04d}"

        pos = PaperPosition(
            ticker=sig.ticker,
            direction=sig.direction.value,
            entry_price=price,
            contracts=sig.contracts,
            opened_at=datetime.now(timezone.utc),
            position_id=pos_id,
        )
        self._positions[sig.ticker] = pos

        log.info(
            "ENTRY | %s | %s | price=%.1fc | z=%.3f | contracts=%d | id=%s",
            sig.direction.value, sig.ticker, price, sig.z_score,
            sig.contracts, pos_id,
        )

    def _process_exit(self, sig: Signal) -> None:
        """Close a paper position from an exit signal."""
        pos = self._positions.get(sig.ticker)
        if pos is None:
            return

        exit_price = self._prices.get(sig.ticker, pos.entry_price)
        held_s = (datetime.now(timezone.utc) - pos.opened_at).total_seconds()

        # Compute PnL
        entry_c = max(1, min(99, int(round(pos.entry_price))))
        exit_c = max(1, min(99, int(round(exit_price))))

        if pos.direction == "BUY_YES":
            gross = (exit_price - pos.entry_price) * pos.contracts
        else:
            gross = (pos.entry_price - exit_price) * pos.contracts

        fees = round_trip_fee(entry_c, exit_c, pos.contracts, maker=self.cfg.assume_maker)
        net = gross - fees

        trade = TradeRecord(
            ticker=pos.ticker,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            contracts=pos.contracts,
            gross_pnl=gross,
            fees=fees,
            net_pnl=net,
            held_seconds=held_s,
            exit_reason=sig.direction.value,
        )
        self._trades.append(trade)
        del self._positions[sig.ticker]

        # Color-code the output
        pnl_sign = "+" if net >= 0 else ""
        log.info(
            "EXIT  | %s | %s | entry=%.1fc exit=%.1fc | gross=%s%.1fc fees=%.0fc net=%s%.1fc | held=%.0fs",
            trade.exit_reason, trade.ticker,
            trade.entry_price, trade.exit_price,
            "+" if gross >= 0 else "", gross, fees,
            pnl_sign, net, held_s,
        )

    # ── Status Reporting ─────────────────────────────────────────────────

    def _print_config(self) -> None:
        """Print active configuration."""
        log.info("Config: signal_threshold=%.2f cooldown=%ds price_range=[%.0f, %.0f]c",
                 self.cfg.signal_threshold, self.cfg.cooldown_s,
                 self.cfg.price_filter_min, self.cfg.price_filter_max)
        log.info("Config: profit_target=%.1fc stop_loss=%.1fc trailing_stop=%.1fc",
                 self.cfg.profit_target_cents, self.cfg.stop_loss_cents,
                 self.cfg.trailing_stop_cents)
        log.info("Config: dfm_vol_threshold=%.1f momentum_filter=%s assume_maker=%s",
                 self.cfg.dfm_vol_threshold, self.cfg.momentum_filter_enabled,
                 self.cfg.assume_maker)
        log.info("Config: eval_interval=%ds min_obs=%d", EVAL_INTERVAL_S, MIN_OBS)

    def _print_status(self) -> None:
        """Print current session status."""
        elapsed = time.monotonic() - self._start_time
        elapsed_min = elapsed / 60.0

        n_trades = len(self._trades)
        total_pnl = sum(t.net_pnl for t in self._trades)
        total_fees = sum(t.fees for t in self._trades)
        wins = sum(1 for t in self._trades if t.net_pnl > 0)
        win_rate = wins / n_trades if n_trades > 0 else 0.0

        n_open = len(self._positions)

        # Unrealized PnL on open positions
        unrealized = 0.0
        for pos in self._positions.values():
            current = self._prices.get(pos.ticker, pos.entry_price)
            if pos.direction == "BUY_YES":
                unrealized += (current - pos.entry_price) * pos.contracts
            else:
                unrealized += (pos.entry_price - current) * pos.contracts

        log.info("-" * 70)
        log.info(
            "STATUS | %.1fmin | tickers=%d ticks=%d | trades=%d wins=%d (%.0f%%) | "
            "realized=%+.1fc fees=%.1fc | open=%d unrealized=%+.1fc",
            elapsed_min, len(self._known_tickers), self._tick_count,
            n_trades, wins, win_rate * 100, total_pnl, total_fees,
            n_open, unrealized,
        )

        # Per-trade breakdown if any trades completed
        if self._trades:
            pnls = [t.net_pnl for t in self._trades]
            log.info(
                "PnL | total=%+.1fc mean=%+.2fc best=%+.1fc worst=%+.1fc",
                total_pnl, np.mean(pnls), max(pnls), min(pnls),
            )

        # Open positions
        for pos in self._positions.values():
            current = self._prices.get(pos.ticker, pos.entry_price)
            if pos.direction == "BUY_YES":
                upnl = (current - pos.entry_price) * pos.contracts
            else:
                upnl = (pos.entry_price - current) * pos.contracts
            log.info(
                "OPEN  | %s | %s | entry=%.1fc current=%.1fc | unrealized=%+.1fc",
                pos.direction, pos.ticker, pos.entry_price, current, upnl,
            )
        log.info("-" * 70)

    def print_final_summary(self) -> None:
        """Print final session summary on shutdown."""
        elapsed = time.monotonic() - self._start_time
        n_trades = len(self._trades)

        log.info("=" * 70)
        log.info("  SESSION SUMMARY")
        log.info("=" * 70)
        log.info("  Duration:     %.1f minutes", elapsed / 60.0)
        log.info("  Tickers:      %d", len(self._known_tickers))
        log.info("  Total ticks:  %d", self._tick_count)
        log.info("  Trades:       %d", n_trades)

        if n_trades > 0:
            total_pnl = sum(t.net_pnl for t in self._trades)
            total_fees = sum(t.fees for t in self._trades)
            wins = sum(1 for t in self._trades if t.net_pnl > 0)
            pnls = [t.net_pnl for t in self._trades]

            log.info("  Win rate:     %d/%d (%.1f%%)", wins, n_trades, 100 * wins / n_trades)
            log.info("  Total PnL:    %+.1fc", total_pnl)
            log.info("  Total fees:   %.1fc", total_fees)
            log.info("  Mean PnL:     %+.2fc", np.mean(pnls))
            log.info("  Best trade:   %+.1fc", max(pnls))
            log.info("  Worst trade:  %+.1fc", min(pnls))
            if len(pnls) > 1 and np.std(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
                log.info("  Sharpe:       %.2f", sharpe)

            log.info("")
            log.info("  %-6s %-24s %-8s %7s %7s %7s %7s %6s",
                     "Dir", "Ticker", "Reason", "Entry", "Exit", "Gross", "Net", "Held")
            for t in self._trades:
                log.info("  %-6s %-24s %-8s %6.1fc %6.1fc %+6.1fc %+6.1fc %5.0fs",
                         t.direction[:6], t.ticker[:24], t.exit_reason[:8],
                         t.entry_price, t.exit_price, t.gross_pnl, t.net_pnl,
                         t.held_seconds)

        n_open = len(self._positions)
        if n_open > 0:
            log.info("")
            log.info("  %d position(s) still open (not settled):", n_open)
            for pos in self._positions.values():
                current = self._prices.get(pos.ticker, pos.entry_price)
                log.info("    %s %s entry=%.1fc current=%.1fc",
                         pos.direction, pos.ticker, pos.entry_price, current)

        # Filter funnel stats
        stats = self.signal_gen.get_filter_stats()
        log.info("")
        log.info("  Filter funnel: %s", stats)
        log.info("=" * 70)


# ── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    trader = LivePaperTrader()

    # Handle Ctrl+C gracefully
    def shutdown(signum, frame):
        log.info("Shutting down...")
        trader._running = False
        trader.print_final_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
