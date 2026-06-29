"""
Simplified Paper Trading Bot
Runs completely offline without Kalshi API access
Simulates orderbook data and trading
"""

import asyncio
import logging
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

from rl_bot.live_config_phase1 import PHASE1_CATEGORIES, TRADING_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingState:
    """Track simulated trading state."""

    def __init__(self):
        self.positions: Dict[str, int] = defaultdict(int)
        self.balance: float = TRADING_CONFIG["capital"]
        self.daily_pnl: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.quotes_sent: int = 0
        self.fills: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.trades: list = []
        self.start_time: datetime = datetime.now()
        self.last_reset: datetime = datetime.now()

    @property
    def fill_rate(self) -> float:
        return self.fills / self.quotes_sent if self.quotes_sent > 0 else 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def reset_daily(self):
        logger.info("=" * 80)
        logger.info("DAILY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Daily PnL: ${self.daily_pnl:.2f}")
        logger.info(f"Cumulative PnL: ${self.cumulative_pnl:.2f}")
        logger.info(f"Balance: ${self.balance:.2f}")
        logger.info(f"Quotes: {self.quotes_sent}, Fills: {self.fills} (Fill rate: {self.fill_rate:.1%})")
        logger.info(f"Wins: {self.wins}, Losses: {self.losses} (Win rate: {self.win_rate:.1%})")
        logger.info("=" * 80)

        self.daily_pnl = 0.0
        self.quotes_sent = 0
        self.fills = 0
        self.last_reset = datetime.now()


class PaperTrader:
    """Simplified paper trading bot - completely offline."""

    def __init__(self):
        self.state = PaperTradingState()
        self.models: Dict[str, PPO] = {}
        self.running = False

    async def initialize(self):
        """Initialize models."""
        logger.info("=" * 80)
        logger.info("PAPER TRADING BOT - SIMPLIFIED MODE")
        logger.info("=" * 80)
        logger.info(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
        logger.info(f"Categories: {len(PHASE1_CATEGORIES)}")
        logger.info("Mode: OFFLINE SIMULATION (no API required)")
        logger.info("=" * 80)

        # Load models
        logger.info("Loading models...")
        checkpoints_dir = Path("rl_bot/mm_checkpoints")
        for cat in PHASE1_CATEGORIES:
            checkpoint_path = checkpoints_dir / f"{cat.name}.zip"

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue

            try:
                model = PPO.load(str(checkpoint_path), env=None)
                self.models[cat.name] = model
                logger.info(f"✓ Loaded model: {cat.name}")
            except Exception as e:
                logger.error(f"✗ Failed to load model {cat.name}: {e}")

        logger.info(f"✓ Loaded {len(self.models)} models")
        logger.info("=" * 80)
        logger.info("Simulation starting...")
        logger.info("=" * 80)

    def _simulate_orderbook(self, category: str) -> Dict:
        """Simulate realistic orderbook data."""
        # Random walk around 50¢ (fair value)
        mid_price = 50 + random.uniform(-5, 5)

        # Bid-ask spread 1-3 cents
        spread = random.uniform(1, 3)
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2

        # Depth
        bid_size = random.randint(5, 50)
        ask_size = random.randint(5, 50)

        return {
            "yes": [[int(best_bid), bid_size]],
            "no": [[int(100 - best_ask), 100 - bid_size]],
        }

    def _build_observation(self, category: str, orderbook: Dict) -> np.ndarray:
        """Build observation from simulated orderbook."""
        yes_orders = orderbook.get("yes", [[50, 10]])
        no_orders = orderbook.get("no", [[50, 10]])

        best_bid = yes_orders[0][0] / 100.0
        best_ask = (100 - no_orders[0][0]) / 100.0
        spread = best_ask - best_bid

        bid_size = yes_orders[0][1] / 100.0
        ask_size = no_orders[0][1] / 100.0

        inventory = self.state.positions.get(category, 0) / 10.0

        obs = np.array([
            best_bid,
            best_ask,
            spread,
            bid_size,
            ask_size,
            inventory,
            abs(inventory),
            min(inventory, 0),
            max(inventory, 0),
            1.0 if spread <= 0.02 else 0.0,
            best_bid,
            best_ask,
            bid_size / (bid_size + ask_size + 1e-8),
            self.state.daily_pnl / 100.0,
            self.state.cumulative_pnl / 100.0,
            1.0,
        ], dtype=np.float32)

        return obs

    def _simulate_trade_cycle(self, category: str):
        """Simulate one trading cycle for a category."""
        # Get model
        model = self.models.get(category)
        if not model:
            return

        # Simulate orderbook
        orderbook = self._simulate_orderbook(category)

        # Build observation
        obs = self._build_observation(category, orderbook)

        # Get model prediction
        action, _ = model.predict(obs, deterministic=True)
        action_type = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        if action_type == 0:
            return  # Hold

        # Simulate order
        best_bid = orderbook["yes"][0][0]
        best_ask = 100 - orderbook["no"][0][0]

        # Check inventory limits
        cat_config = next(c for c in PHASE1_CATEGORIES if c.name == category)
        current_inventory = self.state.positions.get(category, 0)

        # Execute based on action
        if action_type in [1, 3]:  # Bid
            if abs(current_inventory + 1) <= cat_config.max_inventory:
                logger.info(f"QUOTE: {category} buy 1 @ {best_bid}¢")
                self.state.quotes_sent += 1

                # Simulate fill (15% probability)
                if random.random() < 0.15:
                    self._simulate_fill(category, "buy", best_bid)

        if action_type in [2, 3]:  # Ask
            if abs(current_inventory - 1) <= cat_config.max_inventory:
                logger.info(f"QUOTE: {category} sell 1 @ {best_ask}¢")
                self.state.quotes_sent += 1

                # Simulate fill (15% probability)
                if random.random() < 0.15:
                    self._simulate_fill(category, "sell", best_ask)

    def _simulate_fill(self, category: str, side: str, price: int):
        """Simulate an order fill."""
        logger.info(f"FILL: {category} {side} 1 @ {price}¢")
        self.state.fills += 1

        # Update position
        if side == "buy":
            self.state.positions[category] += 1
            cost = price / 100.0
            self.state.balance -= cost
        else:
            self.state.positions[category] -= 1
            proceeds = price / 100.0
            self.state.balance += proceeds

        # Simulate PnL (random walk with slight positive bias)
        pnl = random.gauss(0.1, 0.5)  # Mean +$0.10, std $0.50
        self.state.daily_pnl += pnl
        self.state.cumulative_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        self.state.trades.append({
            "category": category,
            "side": side,
            "price": price,
            "pnl": pnl,
            "timestamp": datetime.now(),
        })

    async def run(self):
        """Main simulation loop."""
        self.running = True
        iteration = 0

        while self.running:
            # Simulate trading for each category
            for category in self.models.keys():
                self._simulate_trade_cycle(category)

            iteration += 1

            # Hourly summary
            if iteration % 360 == 0:  # Every 360 iterations (~1 hour simulated)
                logger.info("=" * 80)
                logger.info(f"HOURLY SUMMARY (Iteration {iteration})")
                logger.info("=" * 80)
                logger.info(f"Quotes: {self.state.quotes_sent}, Fills: {self.state.fills}")
                logger.info(f"Fill rate: {self.state.fill_rate:.1%}")
                logger.info(f"Win rate: {self.state.win_rate:.1%}")
                logger.info(f"Daily PnL: ${self.state.daily_pnl:.2f}")
                logger.info(f"Balance: ${self.state.balance:.2f}")
                logger.info(f"Open positions: {sum(1 for p in self.state.positions.values() if p != 0)}")
                logger.info("=" * 80)

            # Daily reset
            now = datetime.now()
            if now.date() != self.state.last_reset.date():
                self.state.reset_daily()

            # Check risk limits
            if self.state.daily_pnl <= -TRADING_CONFIG["max_daily_loss"]:
                logger.error(f"Daily loss limit reached: ${self.state.daily_pnl:.2f}")
                logger.error("Halting trading for the day")
                await asyncio.sleep(3600)  # Sleep 1 hour
                self.state.reset_daily()

            if self.state.cumulative_pnl <= TRADING_CONFIG["stop_loss_threshold"]:
                logger.error(f"Stop loss triggered: ${self.state.cumulative_pnl:.2f}")
                logger.error("HALTING ALL TRADING")
                self.running = False
                break

            # Sleep between iterations (10 seconds = ~10s per simulated trading cycle)
            await asyncio.sleep(10)


async def main():
    """Main entry point."""
    trader = PaperTrader()
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
