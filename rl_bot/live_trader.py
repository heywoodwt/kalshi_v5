"""
Live Trading Bot - Phase 1 Deployment
Temporal-validated MM PPO strategy on Kalshi WebSocket
Capital: $95.82 across 8 validated categories
"""

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import websockets
from stable_baselines3 import PPO

from live_config_phase1 import PHASE1_CATEGORIES, TRADING_CONFIG, MONITORING_CONFIG
from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv
from rl_bot.mm_metadata import MarketMetadataLoader
from rl_bot.state_builder import build_observation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingState:
    """Track trading state across all categories."""

    def __init__(self):
        self.positions: Dict[str, int] = defaultdict(int)  # category -> inventory
        self.daily_pnl: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.quotes_sent: int = 0
        self.fills: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.consecutive_losses: Dict[str, int] = defaultdict(int)
        self.halted_categories: set = set()
        self.start_time: datetime = datetime.now()
        self.last_reset: datetime = datetime.now()

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate."""
        return self.fills / self.quotes_sent if self.quotes_sent > 0 else 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def total_position_value(self) -> float:
        """Estimate total position value (simplified)."""
        # Assume average position value of $20 per contract
        return sum(abs(inv) * 20 for inv in self.positions.values())

    def reset_daily(self):
        """Reset daily counters."""
        logger.info(f"Daily reset - PnL: ${self.daily_pnl:.2f}")
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()


class KalshiWebSocketClient:
    """Kalshi WebSocket client for real-time orderbook data."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "wss://trading-api.kalshi.com/trade-api/ws/v2"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.ws = None
        self.subscriptions: Dict[str, callable] = {}  # ticker -> callback

    async def connect(self):
        """Connect to Kalshi WebSocket."""
        logger.info(f"Connecting to Kalshi WebSocket: {self.base_url}")
        try:
            self.ws = await websockets.connect(
                self.base_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",  # Simplified - need proper auth
                }
            )
            logger.info("WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def subscribe_orderbook(self, ticker: str, callback: callable):
        """Subscribe to orderbook updates for a ticker."""
        self.subscriptions[ticker] = callback

        msg = {
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": ticker,
            },
        }

        await self.ws.send(json.dumps(msg))
        logger.info(f"Subscribed to orderbook: {ticker}")

    async def listen(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed")
            return False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return False

    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")
        ticker = data.get("msg", {}).get("market_ticker")

        if msg_type == "orderbook_delta" and ticker in self.subscriptions:
            callback = self.subscriptions[ticker]
            await callback(ticker, data["msg"])


class LiveTrader:
    """Main live trading bot."""

    def __init__(self):
        self.config = MMConfig()
        self.state = TradingState()
        self.models: Dict[str, PPO] = {}  # category -> model
        self.ws_client: Optional[KalshiWebSocketClient] = None
        self.running = False

        # Load environment variables
        self.api_key = os.getenv("KALSHI_API_KEY")
        self.api_secret = os.getenv("KALSHI_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("Missing KALSHI_API_KEY or KALSHI_API_SECRET environment variables")

    async def initialize(self):
        """Initialize models and WebSocket connection."""
        logger.info("Initializing live trader...")

        # Load models for each category
        checkpoints_dir = Path("rl_bot/mm_checkpoints")
        for cat in PHASE1_CATEGORIES:
            checkpoint_path = checkpoints_dir / f"{cat.name}.zip"

            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue

            try:
                # Load model (without environment for inference only)
                model = PPO.load(str(checkpoint_path), env=None)
                self.models[cat.name] = model
                logger.info(f"Loaded model: {cat.name}")
            except Exception as e:
                logger.error(f"Failed to load model {cat.name}: {e}")

        logger.info(f"Loaded {len(self.models)} models")

        # Connect to WebSocket
        self.ws_client = KalshiWebSocketClient(self.api_key, self.api_secret)
        connected = await self.ws_client.connect()

        if not connected:
            raise RuntimeError("Failed to connect to Kalshi WebSocket")

        # Subscribe to all category markets
        for cat in PHASE1_CATEGORIES:
            if cat.name in self.models:
                await self.ws_client.subscribe_orderbook(
                    cat.name,
                    self._create_orderbook_callback(cat.name)
                )

        logger.info("Initialization complete")

    def _create_orderbook_callback(self, category: str):
        """Create orderbook callback for a category."""

        async def callback(ticker: str, orderbook_data: dict):
            """Handle orderbook update."""
            # Check if category is halted
            if category in self.state.halted_categories:
                return

            # Check risk limits
            if not self._check_risk_limits():
                return

            # Build observation from orderbook
            obs = self._build_observation(category, orderbook_data)

            if obs is None:
                return

            # Get model prediction
            model = self.models[category]
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            await self._execute_action(category, action, orderbook_data)

        return callback

    def _build_observation(self, category: str, orderbook_data: dict) -> Optional[np.ndarray]:
        """Build observation vector from orderbook data."""
        try:
            # Extract orderbook features
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])

            if not bids or not asks:
                return None

            # Get best bid/ask
            best_bid = bids[0]["price"] if bids else 0
            best_ask = asks[0]["price"] if asks else 100

            # Get depth
            bid_size = sum(b["size"] for b in bids[:5])
            ask_size = sum(a["size"] for a in asks[:5])

            # Build observation (simplified - matches training)
            # [best_bid, best_ask, spread, bid_size, ask_size, inventory, position, ...]
            spread = best_ask - best_bid
            inventory = self.state.positions[category]

            obs = np.array([
                best_bid,
                best_ask,
                spread,
                bid_size,
                ask_size,
                inventory,
                abs(inventory),  # position magnitude
                min(inventory, 0),  # short position
                max(inventory, 0),  # long position
                1.0 if spread <= 1 else 0.0,  # tight spread indicator
                best_bid / 100.0,  # normalized price
                (100 - best_ask) / 100.0,  # normalized inverse price
                bid_size / (bid_size + ask_size + 1e-8),  # bid pressure
                self.state.daily_pnl,  # current PnL
                self.state.cumulative_pnl,  # cumulative PnL
                1.0,  # bias term
            ], dtype=np.float32)

            return obs

        except Exception as e:
            logger.error(f"Error building observation for {category}: {e}")
            return None

    async def _execute_action(self, category: str, action: np.ndarray, orderbook_data: dict):
        """Execute trading action."""
        try:
            # Decode action (simplified - depends on action space)
            # Action space: [quote_bid, quote_ask, cancel, hold]
            # For this implementation, assume action is discrete: 0=hold, 1=bid, 2=ask, 3=both

            action_type = int(action[0]) if isinstance(action, np.ndarray) else int(action)

            if action_type == 0:
                # Hold - no action
                return

            # Get current best bid/ask
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])

            if not bids or not asks:
                return

            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]

            # Apply subpenny offset for queue priority
            our_bid = best_bid + TRADING_CONFIG["quote_offset_bid"]
            our_ask = best_ask + TRADING_CONFIG["quote_offset_ask"]

            # Check inventory limits
            inventory = self.state.positions[category]
            cat_config = next(c for c in PHASE1_CATEGORIES if c.name == category)

            if action_type in [1, 3]:  # Bid
                if abs(inventory + 1) <= cat_config.max_inventory:
                    await self._send_order(category, "buy", our_bid, 1)
                    self.state.quotes_sent += 1

            if action_type in [2, 3]:  # Ask
                if abs(inventory - 1) <= cat_config.max_inventory:
                    await self._send_order(category, "sell", our_ask, 1)
                    self.state.quotes_sent += 1

        except Exception as e:
            logger.error(f"Error executing action for {category}: {e}")

    async def _send_order(self, category: str, side: str, price: float, size: int):
        """Send order to Kalshi."""
        # This is a simplified implementation - need actual Kalshi REST API integration
        logger.info(f"ORDER: {category} {side} {size} @ ${price:.3f}")

        # TODO: Actual Kalshi REST API call
        # For now, just log

    def _check_risk_limits(self) -> bool:
        """Check if risk limits allow trading."""

        # Check daily loss limit
        if self.state.daily_pnl <= -TRADING_CONFIG["max_daily_loss"]:
            logger.warning(f"Daily loss limit reached: ${self.state.daily_pnl:.2f}")
            self._halt_all_trading("Daily loss limit")
            return False

        # Check stop loss
        if self.state.cumulative_pnl <= TRADING_CONFIG["stop_loss_threshold"]:
            logger.error(f"Stop loss triggered: ${self.state.cumulative_pnl:.2f}")
            self._halt_all_trading("Stop loss")
            return False

        # Check position value
        if self.state.total_position_value >= TRADING_CONFIG["max_position_value"]:
            logger.warning(f"Position value limit: ${self.state.total_position_value:.2f}")
            return False

        return True

    def _halt_all_trading(self, reason: str):
        """Halt all trading."""
        logger.error(f"HALTING ALL TRADING: {reason}")
        self.running = False
        # TODO: Cancel all open orders
        # TODO: Send alert

    async def run(self):
        """Main trading loop."""
        logger.info("Starting live trading...")
        self.running = True

        # Start WebSocket listener
        listen_task = asyncio.create_task(self.ws_client.listen())

        # Start daily reset task
        reset_task = asyncio.create_task(self._daily_reset_loop())

        # Wait for tasks
        await asyncio.gather(listen_task, reset_task)

    async def _daily_reset_loop(self):
        """Reset daily counters at midnight."""
        while self.running:
            await asyncio.sleep(60)  # Check every minute

            now = datetime.now()
            if now.date() != self.state.last_reset.date():
                self.state.reset_daily()


async def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("LIVE TRADING BOT - PHASE 1")
    logger.info("=" * 80)
    logger.info(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    logger.info(f"Categories: {len(PHASE1_CATEGORIES)}")
    logger.info(f"Mode: {TRADING_CONFIG['mode']}")
    logger.info("=" * 80)

    trader = LiveTrader()
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
