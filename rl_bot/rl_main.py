"""RL Trading Bot — Main Entry Point

Runs a Dueling DQN agent that trades Kalshi BTC binary contracts.
Connects to the live Kalshi WebSocket for market data and polls
external exchanges for BTC spot/funding rate features.

Usage:
    python -m rl_bot.rl_main              # paper trading (default)
    PAPER_TRADING=false python -m rl_bot.rl_main  # live trading

The agent runs parallel to the existing TFT/HP-DFM pipeline.
"""
import asyncio
import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from authentication_to_kalshi.websocket_client import KalshiWebSocket
from market_selector.market_filter import is_btc_market, register_market
from model.hp_dfm_rte.orderbook import extract_orderbook_snapshot
from rl_bot.agent import DQNAgent
from rl_bot.btc_data import BTCDataPoller
from rl_bot.config import RLConfig, decode_action
from rl_bot.environment import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rl_bot")


def _parse_expiry_hours(ticker: str) -> float:
    """Estimate hours to expiry from ticker name.

    Kalshi BTC tickers look like KXBTC-26JUN21-T17 (date + hour).
    Returns a rough estimate; defaults to 4.0 if parsing fails.
    """
    try:
        parts = ticker.split("-")
        if len(parts) < 3:
            return 4.0
        # Extract the date part (e.g., "26JUN21") and hour part (e.g., "T17")
        date_str = parts[1]
        hour_part = parts[2] if len(parts) > 2 else ""
        # Parse date: YYMMMDD or DDMMMYY format varies
        # For now, just use a rough estimate based on the hour tag
        if hour_part.startswith("T") and hour_part[1:].isdigit():
            target_hour = int(hour_part[1:])
            now = datetime.now(timezone.utc)
            # Rough hours until that hour today or tomorrow
            hours_diff = target_hour - now.hour
            if hours_diff < 0:
                hours_diff += 24
            return max(0.1, float(hours_diff))
    except Exception:
        pass
    return 4.0


class RLBot:
    """Orchestrates the RL agent with the Kalshi WebSocket feed.

    Wires together: WebSocket callbacks -> TradingEnv -> DQNAgent -> logging.
    Runs the agent decision loop on a fixed 30-second cadence.
    """

    def __init__(self, cfg: RLConfig) -> None:
        self._cfg = cfg
        self._btc = BTCDataPoller(poll_interval_s=cfg.decision_interval_s)
        self._env = TradingEnv(cfg, self._btc)
        self._agent = DQNAgent(cfg)

        # Previous state per market (for storing transitions)
        self._prev_states: dict[str, tuple] = {}  # ticker -> (state, action, mask)

        # CSV logger
        self._csv_path = cfg.log_csv_path
        self._csv_initialized = False

    # -- WebSocket callbacks (called from the WS receive loop) --

    def on_ticker(self, msg: dict) -> None:
        """Handle ticker message from Kalshi WebSocket."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not is_btc_market(ticker):
            return

        if register_market(ticker):
            log.info("RL bot discovered market: %s", ticker)

        # Extract and normalize price
        raw_price = data.get("yes_ask") or data.get("last_price")
        if raw_price is None:
            return
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            return
        price = price / 100 if price > 1 else price

        # Estimate time to expiry from ticker name
        tte = _parse_expiry_hours(ticker)

        self._env.on_ticker(ticker, price, tte)

    def on_trade(self, msg: dict) -> None:
        """Handle trade message from Kalshi WebSocket."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not is_btc_market(ticker):
            return

        price = data.get("yes_price", 0)
        price = price / 100 if price > 1 else price
        count = data.get("count", 1)

        self._env.on_trade(ticker, price, count)

    def on_orderbook_delta(self, msg: dict) -> None:
        """Handle orderbook delta message from Kalshi WebSocket."""
        data = msg.get("msg", {})
        ticker = data.get("market_ticker", "")
        if not is_btc_market(ticker):
            return

        snapshot = extract_orderbook_snapshot(data)
        if snapshot is not None:
            self._env.on_orderbook(ticker, snapshot)

    # -- Decision loop --

    async def decision_loop(self) -> None:
        """Runs every decision_interval_s seconds. Evaluates all active markets."""
        while True:
            await asyncio.sleep(self._cfg.decision_interval_s)

            markets = self._env.get_active_markets()
            if not markets:
                continue

            circuit_active = self._env.is_circuit_breaker_active()

            for ticker in markets:
                state = self._env.get_state(ticker)
                mask = self._env.get_mask(ticker)

                # If circuit breaker is active, only allow HOLD and CLOSE
                if circuit_active:
                    for i in range(18):  # mask all buy actions
                        mask[i] = 0.0

                # Store transition from previous step (if any)
                if ticker in self._prev_states:
                    prev_state, prev_action, _ = self._prev_states[ticker]
                    # The reward from the previous action is embedded in the
                    # environment's PnL tracker. For HOLD/BUY opens, reward=0.
                    # We use 0 here; close/settle rewards are handled at close time.
                    self._agent.store_transition(
                        state=prev_state,
                        action=prev_action,
                        reward=0.0,  # deferred reward model
                        next_state=state,
                        done=False,
                    )

                # Select action
                action = self._agent.select_action(state, mask)

                # Execute action
                next_state, reward, done = self._env.step(ticker, action)

                # If we got a non-zero reward (close/settle), store that transition
                if reward != 0.0:
                    self._agent.store_transition(
                        state=state, action=action, reward=reward,
                        next_state=next_state, done=done,
                    )
                    self._prev_states.pop(ticker, None)
                else:
                    # Save for deferred reward on next step
                    self._prev_states[ticker] = (state, action, mask)

                # Log the decision
                self._log_step(ticker, action, state, reward)

            # Train one step
            self._agent.step_count += 1
            loss = self._agent.train_step()
            if loss is not None:
                log.debug("Train step %d, loss=%.6f, eps=%.3f",
                          self._agent.step_count, loss, self._agent.epsilon())

            # Checkpoint periodically
            if self._agent.step_count % self._cfg.checkpoint_freq == 0:
                self._save_checkpoint()

    def _log_step(self, ticker: str, action: int, state, reward: float) -> None:
        """Log a decision step to stdout and CSV."""
        decoded = decode_action(action)
        if isinstance(decoded, tuple):
            action_str = f"BUY_{decoded[0].upper()}_{decoded[1]}_AT_{int(decoded[2]*100)}c"
        else:
            action_str = decoded.upper()

        position = self._env.pnl_tracker.get_position(ticker)
        daily = self._env.pnl_tracker.daily_pnl()
        eps = self._agent.epsilon()

        log.info(
            "[%s] action=%s pos=%d reward=%.4f daily_pnl=%.4f eps=%.3f step=%d",
            ticker, action_str, position, reward, daily, eps, self._agent.step_count,
        )

        # CSV logging
        self._write_csv_row(ticker, action_str, eps, position, reward, daily)

    def _write_csv_row(
        self, ticker: str, action: str, eps: float,
        position: int, reward: float, daily_pnl: float,
    ) -> None:
        """Append a row to the CSV log file."""
        # Ensure output directory exists
        Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if self._csv_initialized else "w"
        with open(self._csv_path, mode, newline="") as f:
            writer = csv.writer(f)
            if not self._csv_initialized:
                writer.writerow([
                    "timestamp", "ticker", "action", "epsilon",
                    "position", "reward", "cumulative_pnl", "step",
                ])
                self._csv_initialized = True
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                ticker, action, f"{eps:.4f}",
                position, f"{reward:.6f}", f"{daily_pnl:.6f}",
                self._agent.step_count,
            ])

    def _save_checkpoint(self) -> None:
        """Save agent checkpoint to disk."""
        ckpt_dir = Path(self._cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = str(ckpt_dir / f"rl_agent_step_{self._agent.step_count}.pt")
        self._agent.save_checkpoint(path)
        log.info("Checkpoint saved: %s", path)

    def _load_latest_checkpoint(self) -> None:
        """Load the most recent checkpoint if available."""
        ckpt_dir = Path(self._cfg.checkpoint_dir)
        if not ckpt_dir.exists():
            return
        # Find latest checkpoint by step number
        ckpts = sorted(ckpt_dir.glob("rl_agent_step_*.pt"))
        if ckpts:
            latest = str(ckpts[-1])
            self._agent.load_checkpoint(latest)
            log.info("Loaded checkpoint: %s (step %d)", latest, self._agent.step_count)


async def main() -> None:
    """Main entry point for the RL trading bot."""
    # Load config
    paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
    cfg = RLConfig(paper_trading=paper)

    mode = "PAPER" if cfg.paper_trading else "LIVE"
    log.info("Starting RL trading bot (%s mode)", mode)
    log.info("Config: %s", cfg)

    # Create bot
    bot = RLBot(cfg)
    bot._load_latest_checkpoint()

    # Create WebSocket client with RL bot's callbacks
    ws = KalshiWebSocket(
        on_ticker=bot.on_ticker,
        on_trade=bot.on_trade,
        on_orderbook_delta=bot.on_orderbook_delta,
    )

    # Run WebSocket + BTC poller + decision loop concurrently
    await asyncio.gather(
        ws.run(),               # WebSocket connection loop (never returns)
        bot._btc.start(),       # BTC spot/funding poller (never returns)
        bot.decision_loop(),    # Agent decision loop (never returns)
    )


if __name__ == "__main__":
    asyncio.run(main())
