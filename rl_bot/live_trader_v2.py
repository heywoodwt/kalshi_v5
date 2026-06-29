"""
Live Trading Bot - Phase 1 with Full Kalshi API Integration
Temporal-validated MM PPO strategy
Capital: $95.82 across 8 validated categories
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Literal

import numpy as np
import websockets
from stable_baselines3 import PPO

from rl_bot.kalshi_api import KalshiRESTClient, KalshiPaperTradingClient, KalshiAPIError
from rl_bot.live_config_phase1 import PHASE1_CATEGORIES, TRADING_CONFIG, MONITORING_CONFIG
from rl_bot.mm_config import MMConfig

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
        self.positions: Dict[str, int] = defaultdict(int)  # ticker -> inventory
        self.daily_pnl: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.quotes_sent: int = 0
        self.fills: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.trades: list = []  # Track all trades for PnL calculation
        self.consecutive_losses: Dict[str, int] = defaultdict(int)
        self.halted_categories: set = set()
        self.pending_orders: Dict[str, Dict] = {}  # order_id -> order info
        self.start_time: datetime = datetime.now()
        self.last_reset: datetime = datetime.now()
        self.last_fill_check: datetime = datetime.now()

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

    def record_trade(self, ticker: str, side: str, price: float, size: int):
        """Record a trade for PnL tracking."""
        self.trades.append({
            "timestamp": datetime.now(),
            "ticker": ticker,
            "side": side,
            "price": price,
            "size": size,
        })

    def calculate_pnl(self, ticker: str, last_price: float) -> float:
        """Calculate unrealized + realized PnL for a ticker."""
        # Simplified - would need proper mark-to-market
        position = self.positions[ticker]
        if position == 0:
            return 0.0

        # Estimate PnL based on position and current price
        # This is simplified - real implementation needs entry prices
        return position * (last_price - 50) / 100.0  # Rough estimate

    def reset_daily(self):
        """Reset daily counters."""
        logger.info(f"Daily reset - PnL: ${self.daily_pnl:.2f}, Cumulative: ${self.cumulative_pnl:.2f}")
        logger.info(f"Fill rate: {self.fill_rate:.1%}, Win rate: {self.win_rate:.1%}")
        self.daily_pnl = 0.0
        self.quotes_sent = 0
        self.fills = 0
        self.last_reset = datetime.now()


class LiveTrader:
    """Main live trading bot with full Kalshi API integration."""

    def __init__(self, paper_mode: bool = False):
        self.paper_mode = paper_mode
        self.config = MMConfig()
        self.state = TradingState()
        self.models: Dict[str, PPO] = {}  # category -> model
        self.ws_client: Optional['KalshiWebSocketClient'] = None
        self.api_client: Optional[KalshiRESTClient] = None
        self.running = False
        self.active_tickers: Dict[str, str] = {}  # category -> current ticker

        # Load environment variables
        self.api_key = os.getenv("KALSHI_API_KEY")
        self.api_secret = os.getenv("KALSHI_API_SECRET")

        if not self.api_key or not self.api_secret:
            logger.warning("Missing KALSHI_API_KEY or KALSHI_API_SECRET - using demo mode")
            self.paper_mode = True

    async def initialize(self):
        """Initialize models, API client, and WebSocket connection."""
        logger.info("=" * 80)
        logger.info(f"Initializing live trader (Mode: {'PAPER' if self.paper_mode else 'LIVE'})...")
        logger.info("=" * 80)

        # Initialize API client
        if self.paper_mode:
            logger.info("Using PAPER TRADING mode")
            self.api_client = KalshiPaperTradingClient(
                api_key=self.api_key or "paper_key",
                api_secret=self.api_secret or "paper_secret",
                initial_balance=TRADING_CONFIG["capital"]
            )
        else:
            logger.info("Using LIVE TRADING mode")
            self.api_client = KalshiRESTClient(
                api_key=self.api_key,
                api_secret=self.api_secret
            )

            # Login and get token
            email = os.getenv("KALSHI_EMAIL")
            password = os.getenv("KALSHI_PASSWORD")
            if email and password:
                logger.info("Logging in to Kalshi...")
                self.api_client.login(email, password)

        # Load models for each category
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

        logger.info(f"Loaded {len(self.models)} models")

        # Get active tickers for each category
        logger.info("Finding active markets...")
        for cat in PHASE1_CATEGORIES:
            if cat.name in self.models:
                ticker = await self._find_active_ticker(cat.name)
                if ticker:
                    self.active_tickers[cat.name] = ticker
                    logger.info(f"✓ {cat.name}: {ticker}")
                else:
                    logger.warning(f"✗ No active market for {cat.name}")

        # Connect to WebSocket
        logger.info("Connecting to WebSocket...")
        self.ws_client = KalshiWebSocketClient(
            api_key=self.api_key or "demo",
            api_secret=self.api_secret or "demo"
        )
        connected = await self.ws_client.connect()

        if not connected:
            raise RuntimeError("Failed to connect to Kalshi WebSocket")

        # Subscribe to all active tickers
        for category, ticker in self.active_tickers.items():
            await self.ws_client.subscribe_orderbook(
                ticker,
                self._create_orderbook_callback(category, ticker)
            )

        logger.info("=" * 80)
        logger.info("Initialization complete - ready to trade!")
        logger.info("=" * 80)

    async def _find_active_ticker(self, category: str) -> Optional[str]:
        """Find an active (open) market ticker for a category."""
        try:
            # Search for markets with this series ticker
            response = self.api_client.get_markets(
                series_ticker=category,
                status="open",
                limit=1
            )

            markets = response.get("markets", [])
            if markets:
                return markets[0]["ticker"]

            return None

        except KalshiAPIError as e:
            logger.error(f"Error finding ticker for {category}: {e}")
            return None

    def _create_orderbook_callback(self, category: str, ticker: str):
        """Create orderbook callback for a category."""

        async def callback(ticker_msg: str, orderbook_data: dict):
            """Handle orderbook update."""
            # Check if category is halted
            if category in self.state.halted_categories:
                return

            # Check risk limits
            if not self._check_risk_limits():
                return

            # Build observation from orderbook
            obs = self._build_observation(category, ticker, orderbook_data)

            if obs is None:
                return

            # Get model prediction
            model = self.models[category]
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            await self._execute_action(category, ticker, action, orderbook_data)

        return callback

    def _build_observation(self, category: str, ticker: str, orderbook_data: dict) -> Optional[np.ndarray]:
        """Build observation vector from orderbook data."""
        try:
            # Extract orderbook features
            bids = orderbook_data.get("yes", [])  # Kalshi uses yes/no instead of bids/asks
            asks = orderbook_data.get("no", [])

            if not bids or not asks:
                return None

            # Get best bid/ask (prices are in cents, 1-99)
            best_bid = bids[0][0] if bids else 1  # [price, size]
            best_ask = asks[0][0] if asks else 99

            # Get depth
            bid_size = sum(b[1] for b in bids[:5])
            ask_size = sum(a[1] for a in asks[:5])

            # Get current inventory
            inventory = self.state.positions.get(ticker, 0)

            # Build observation (16-dimensional, matching training)
            spread = best_ask - best_bid

            obs = np.array([
                best_bid / 100.0,  # Normalized best bid
                best_ask / 100.0,  # Normalized best ask
                spread / 100.0,  # Normalized spread
                bid_size / 100.0,  # Normalized bid size
                ask_size / 100.0,  # Normalized ask size
                inventory / 10.0,  # Normalized inventory
                abs(inventory) / 10.0,  # Position magnitude
                min(inventory, 0) / 10.0,  # Short position
                max(inventory, 0) / 10.0,  # Long position
                1.0 if spread <= 2 else 0.0,  # Tight spread indicator
                (best_bid / 100.0),  # Bid level
                (best_ask / 100.0),  # Ask level
                bid_size / (bid_size + ask_size + 1e-8),  # Bid pressure
                self.state.daily_pnl / 100.0,  # Normalized daily PnL
                self.state.cumulative_pnl / 100.0,  # Normalized cumulative PnL
                1.0,  # Bias term
            ], dtype=np.float32)

            return obs

        except Exception as e:
            logger.error(f"Error building observation for {category}/{ticker}: {e}")
            return None

    async def _execute_action(self, category: str, ticker: str, action: np.ndarray, orderbook_data: dict):
        """Execute trading action via Kalshi API."""
        try:
            # Decode action (assuming discrete action space: 0=hold, 1=bid, 2=ask, 3=both)
            action_type = int(action[0]) if isinstance(action, np.ndarray) else int(action)

            if action_type == 0:
                # Hold - no action
                return

            # Get current best bid/ask
            yes_orders = orderbook_data.get("yes", [])
            no_orders = orderbook_data.get("no", [])

            if not yes_orders or not no_orders:
                return

            best_yes_bid = yes_orders[0][0] if yes_orders else 1
            best_no_bid = no_orders[0][0] if no_orders else 1

            # Apply subpenny offset for queue priority
            # Subpenny = +0.001 cents (1 cent = 100 basis points, so 0.1 basis points)
            # For Kalshi, prices are in cents, so we can't do true subpenny
            # Instead, we'll just match the best price for queue position
            our_bid_price = best_yes_bid  # Match best price
            our_ask_price = best_no_bid  # Match best price

            # Check inventory limits
            cat_config = next(c for c in PHASE1_CATEGORIES if c.name == category)
            current_inventory = self.state.positions.get(ticker, 0)

            # Execute based on action
            if action_type in [1, 3]:  # Bid (buy)
                if abs(current_inventory + 1) <= cat_config.max_inventory:
                    await self._send_order(
                        category=category,
                        ticker=ticker,
                        side="buy",
                        price=our_bid_price,
                        size=1
                    )
                    self.state.quotes_sent += 1

            if action_type in [2, 3]:  # Ask (sell)
                if abs(current_inventory - 1) <= cat_config.max_inventory:
                    await self._send_order(
                        category=category,
                        ticker=ticker,
                        side="sell",
                        price=our_ask_price,
                        size=1
                    )
                    self.state.quotes_sent += 1

        except Exception as e:
            logger.error(f"Error executing action for {category}/{ticker}: {e}")

    async def _send_order(self, category: str, ticker: str, side: Literal["buy", "sell"],
                          price: int, size: int):
        """Send order to Kalshi via REST API."""
        try:
            logger.info(f"ORDER: {category}/{ticker} {side} {size} @ {price}¢")

            # Place order via Kalshi API
            response = self.api_client.place_limit_order(
                ticker=ticker,
                side=side,
                price_cents=price,
                size=size
            )

            order_id = response.get("order_id")
            if order_id:
                # Track pending order
                self.state.pending_orders[order_id] = {
                    "category": category,
                    "ticker": ticker,
                    "side": side,
                    "price": price,
                    "size": size,
                    "timestamp": datetime.now(),
                }
                logger.info(f"✓ Order placed: {order_id}")

        except KalshiAPIError as e:
            logger.error(f"✗ Order failed: {e}")

    async def _check_fills_loop(self):
        """Periodically check for fills on pending orders."""
        while self.running:
            await asyncio.sleep(5)  # Check every 5 seconds

            if not self.state.pending_orders:
                continue

            try:
                # Get recent fills
                for order_id in list(self.state.pending_orders.keys()):
                    order_info = self.state.pending_orders[order_id]

                    # Check order status
                    try:
                        response = self.api_client.get_order(order_id)
                        order = response.get("order", {})
                        status = order.get("status")

                        if status == "executed":
                            # Order filled!
                            await self._handle_fill(order_id, order_info, order)
                            del self.state.pending_orders[order_id]

                        elif status == "canceled":
                            # Order canceled
                            logger.info(f"Order canceled: {order_id}")
                            del self.state.pending_orders[order_id]

                    except KalshiAPIError as e:
                        logger.error(f"Error checking order {order_id}: {e}")

            except Exception as e:
                logger.error(f"Error in fill check loop: {e}")

    async def _handle_fill(self, order_id: str, order_info: Dict, order_response: Dict):
        """Handle a filled order."""
        category = order_info["category"]
        ticker = order_info["ticker"]
        side = order_info["side"]
        price = order_info["price"]
        size = order_info.get("size", 1)

        logger.info(f"FILL: {category}/{ticker} {side} {size} @ {price}¢")

        # Update position
        if side == "buy":
            self.state.positions[ticker] += size
        else:  # sell
            self.state.positions[ticker] -= size

        # Update fill count
        self.state.fills += 1

        # Record trade
        self.state.record_trade(ticker, side, price, size)

        # Update PnL (simplified - needs proper accounting)
        if side == "sell" and self.state.positions[ticker] >= 0:
            # Closed a long position - realized profit
            pnl = (price - 50) * size / 100.0  # Rough estimate
            self.state.daily_pnl += pnl
            self.state.cumulative_pnl += pnl

            if pnl > 0:
                self.state.wins += 1
                self.state.consecutive_losses[category] = 0
            else:
                self.state.losses += 1
                self.state.consecutive_losses[category] += 1

    async def _sync_positions(self):
        """Sync positions from Kalshi account."""
        try:
            response = self.api_client.get_positions()
            positions = response.get("positions", [])

            for pos in positions:
                ticker = pos["ticker"]
                position = pos.get("position", 0)
                self.state.positions[ticker] = position

            logger.info(f"Synced {len(positions)} positions from Kalshi")

        except KalshiAPIError as e:
            logger.error(f"Error syncing positions: {e}")

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
        """Halt all trading and cancel orders."""
        logger.error(f"HALTING ALL TRADING: {reason}")
        self.running = False

        # Cancel all pending orders
        try:
            self.api_client.cancel_all_orders()
            logger.info("Canceled all open orders")
        except KalshiAPIError as e:
            logger.error(f"Error canceling orders: {e}")

    async def run(self):
        """Main trading loop."""
        logger.info("Starting live trading...")
        self.running = True

        # Sync initial positions
        await self._sync_positions()

        # Start background tasks
        tasks = [
            asyncio.create_task(self.ws_client.listen()),
            asyncio.create_task(self._check_fills_loop()),
            asyncio.create_task(self._daily_reset_loop()),
            asyncio.create_task(self._hourly_report_loop()),
        ]

        # Wait for tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _daily_reset_loop(self):
        """Reset daily counters at midnight."""
        while self.running:
            await asyncio.sleep(60)  # Check every minute

            now = datetime.now()
            if now.date() != self.state.last_reset.date():
                self.state.reset_daily()

    async def _hourly_report_loop(self):
        """Generate hourly summary reports."""
        while self.running:
            await asyncio.sleep(3600)  # Every hour

            logger.info("=" * 80)
            logger.info("HOURLY SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Quotes sent: {self.state.quotes_sent}")
            logger.info(f"Fills: {self.state.fills} (Fill rate: {self.state.fill_rate:.1%})")
            logger.info(f"Wins: {self.state.wins}, Losses: {self.state.losses} (Win rate: {self.state.win_rate:.1%})")
            logger.info(f"Daily PnL: ${self.state.daily_pnl:.2f}")
            logger.info(f"Cumulative PnL: ${self.state.cumulative_pnl:.2f}")
            logger.info(f"Positions: {len([p for p in self.state.positions.values() if p != 0])} open")
            logger.info("=" * 80)


class KalshiWebSocketClient:
    """Kalshi WebSocket client for real-time orderbook data."""

    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = "wss://trading-api.kalshi.com/trade-api/ws/v2"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.ws = None
        self.subscriptions: Dict[str, callable] = {}  # ticker -> callback

    async def connect(self):
        """Connect to Kalshi WebSocket."""
        logger.info(f"Connecting to WebSocket: {self.base_url}")
        try:
            self.ws = await websockets.connect(
                self.base_url,
                extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                }
            )
            logger.info("✓ WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"✗ WebSocket connection failed: {e}")
            return False

    async def subscribe_orderbook(self, ticker: str, callback: callable):
        """Subscribe to orderbook updates for a ticker."""
        self.subscriptions[ticker] = callback

        msg = {
            "id": int(time.time() * 1000),
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": ticker,
            },
        }

        await self.ws.send(json.dumps(msg))
        logger.info(f"✓ Subscribed to {ticker}")

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


async def main():
    """Main entry point."""
    # Check for paper mode flag
    paper_mode = os.getenv("PAPER_MODE", "false").lower() == "true"

    logger.info("=" * 80)
    logger.info("KALSHI LIVE TRADING BOT - PHASE 1")
    logger.info("=" * 80)
    logger.info(f"Mode: {'PAPER TRADING' if paper_mode else 'LIVE TRADING'}")
    logger.info(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    logger.info(f"Categories: {len(PHASE1_CATEGORIES)}")
    logger.info(f"Risk Limits: Daily loss ${TRADING_CONFIG['max_daily_loss']}, Stop loss ${TRADING_CONFIG['stop_loss_threshold']}")
    logger.info("=" * 80)

    trader = LiveTrader(paper_mode=paper_mode)
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
