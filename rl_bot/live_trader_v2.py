"""
Live Trading Bot - June 2026 Deployment
Active categories based on June 29-30 trading data
Capital: $95.37 / 5 = $19.07 per category
Selection: Top 5 by June trade volume with trained models
"""

import asyncio
import base64
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
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv
from stable_baselines3 import PPO

from rl_bot.kalshi_api import KalshiRESTClient, KalshiPaperTradingClient, KalshiAPIError
from rl_bot.live_config_june import JUNE_TOP5_CATEGORIES, TRADING_CONFIG, MONITORING_CONFIG
from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import scale_action
from rl_bot.mm_metadata import MarketMetadataLoader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Back to INFO - DEBUG was too verbose
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
        # Per-ticker state for accurate observations (matching mm_env.py)
        self.mid_history: Dict[str, list] = defaultdict(list)  # ticker -> last N mid prices
        self.entry_prices: Dict[str, float] = {}  # ticker -> weighted avg entry price
        self.fills_buy: Dict[str, int] = defaultdict(int)  # ticker -> buy fill count
        self.fills_sell: Dict[str, int] = defaultdict(int)  # ticker -> sell fill count
        self.close_times: Dict[str, datetime] = {}  # ticker -> market close time
        self.orderbooks: Dict[str, Dict] = {}  # ticker -> full orderbook snapshot (REST + deltas)

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
        # Kalshi contracts trade $0.01-$0.99, assume average position value of $0.50 per contract
        return sum(abs(inv) * 0.50 for inv in self.positions.values())

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
        # Metadata loader for subpenny tick size validation
        self.metadata_loader: Optional[MarketMetadataLoader] = None
        # Live tick sizes from Kalshi API (ticker -> min step size)
        self._tick_sizes: Dict[str, float] = {}

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

            # Smart order cancellation: cancel orders that ADD to positions,
            # but KEEP orders that would close positions and take profit
            try:
                logger.info("Optimizing orders (canceling inventory-building, keeping profit-taking)...")

                # Get current positions
                positions_response = self.api_client.get_positions()
                positions = {}
                for pos in positions_response.get("positions", []):
                    ticker = pos.get("market_ticker")
                    # Total position = long - short contracts
                    position = pos.get("total_traded", 0)
                    if ticker and position != 0:
                        positions[ticker] = position

                logger.info(f"Found {len(positions)} open positions")

                # Get all open orders
                orders_response = self.api_client.get_orders(status="resting", limit=1000)
                orders = orders_response.get("orders", [])

                canceled_count = 0
                kept_count = 0

                for order in orders:
                    order_id = order.get("order_id")
                    ticker = order.get("market_ticker")
                    side = order.get("side")  # "yes" or "no"

                    current_position = positions.get(ticker, 0)

                    # Determine if order adds to or reduces position
                    # Long position (>0): keep sell orders (reduce), cancel buy orders (add)
                    # Short position (<0): keep buy orders (reduce), cancel sell orders (add)
                    should_keep = False

                    if current_position > 0 and side == "no":  # Long position, sell order
                        should_keep = True
                    elif current_position < 0 and side == "yes":  # Short position, buy order
                        should_keep = True

                    if should_keep:
                        kept_count += 1
                        logger.debug(f"Keeping order {order_id} (closes position in {ticker})")
                    else:
                        try:
                            self.api_client.cancel_order(order_id)
                            canceled_count += 1
                        except Exception as e:
                            logger.debug(f"Could not cancel order {order_id}: {e}")

                logger.info(f"✓ Canceled {canceled_count} orders, kept {kept_count} profit-taking orders")

            except Exception as e:
                logger.warning(f"Could not optimize orders: {e}")

        # Load models for each category
        logger.info("Loading June 2026 models...")
        checkpoints_dir = Path("rl_bot/mm_checkpoints")
        checkpoint_prefix = TRADING_CONFIG.get("checkpoint_prefix", "june")

        for cat in JUNE_TOP5_CATEGORIES:
            # Try june_{CAT}_final.zip first, fall back to mm_{CAT}_{CAT}_final.zip
            checkpoint_path = checkpoints_dir / f"{checkpoint_prefix}_{cat.name}_final.zip"
            if not checkpoint_path.exists():
                # Fall back to HPC naming convention
                checkpoint_path = checkpoints_dir / f"mm_{cat.name}_{cat.name}_final.zip"
            if not checkpoint_path.exists():
                logger.warning(f"No checkpoint found for {cat.name}")
                continue

            try:
                model = PPO.load(str(checkpoint_path), env=None)
                self.models[cat.name] = model
                logger.info(f"✓ Loaded model: {cat.name}")
            except Exception as e:
                logger.error(f"✗ Failed to load model {cat.name}: {e}")

        logger.info(f"Loaded {len(self.models)} models")

        # Load market metadata for subpenny tick size validation
        markets_path = Path("output/rl_all_markets_3mo.parquet")
        if markets_path.exists():
            self.metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=str(markets_path))
            logger.info("✓ Loaded market metadata for subpenny pricing")
        else:
            logger.warning("Market metadata not found — subpenny pricing disabled")

        # Get ALL active tickers for each category
        logger.info("Finding active markets...")
        for cat in JUNE_TOP5_CATEGORIES:
            if cat.name in self.models:
                tickers = await self._find_all_active_tickers(cat.name)
                if tickers:
                    # Store all tickers for this category
                    for ticker in tickers:
                        self.active_tickers[ticker] = cat.name  # ticker -> category mapping
                    logger.info(f"✓ {cat.name}: {len(tickers)} markets")
                else:
                    logger.warning(f"✗ No active market for {cat.name}")

        # Fetch live tick sizes and close times from Kalshi API
        logger.info("Fetching market details from Kalshi API...")
        subpenny_count = 0
        for ticker in list(self.active_tickers.keys()):
            try:
                resp = self.api_client._request('GET', f'/markets/{ticker}')
                market = resp.get('market', {})
                # Tick sizes for subpenny validation
                ranges = market.get('price_ranges', [])
                min_step = min((float(r['step']) for r in ranges), default=0.01)
                self._tick_sizes[ticker] = min_step
                if min_step <= 0.001:
                    subpenny_count += 1
                # Close time for TTE calculation
                close_str = market.get('close_time') or market.get('expected_expiration_time')
                if close_str:
                    # Parse ISO format, strip timezone for naive comparison
                    ct = datetime.fromisoformat(close_str.replace('Z', '+00:00'))
                    self.state.close_times[ticker] = ct
            except Exception:
                self._tick_sizes[ticker] = 0.01
        logger.info(f"✓ Market details loaded: {subpenny_count}/{len(self.active_tickers)} support subpenny")

        # Connect to WebSocket
        logger.info("Connecting to WebSocket...")
        # Try to connect WebSocket (optional - will use REST API polling if fails)
        try:
            self.ws_client = KalshiWebSocketClient(
                api_key=self.api_key or "demo",
                api_secret=self.api_secret or "demo"
            )
            connected = await self.ws_client.connect()

            if connected:
                # Subscribe to WebSocket deltas first (fast)
                for ticker, category in self.active_tickers.items():
                    await self.ws_client.subscribe_orderbook(
                        ticker,
                        self._create_orderbook_callback(category, ticker)
                    )
                logger.info(f"✓ WebSocket enabled - subscribed to {len(self.active_tickers)} markets")

                # Fetch initial orderbook snapshots in background (slow but necessary)
                # This runs async so trading can start immediately on markets with activity
                logger.info(f"Fetching initial orderbook snapshots for {len(self.active_tickers)} markets (background)...")
                asyncio.create_task(self._fetch_initial_snapshots())
            else:
                logger.warning("WebSocket unavailable - using REST API polling")
                self.ws_client = None
        except Exception as e:
            logger.warning(f"WebSocket unavailable ({e}) - using REST API polling")
            self.ws_client = None

        logger.info("=" * 80)
        logger.info("Initialization complete - ready to trade!")
        logger.info("=" * 80)

    async def _fetch_initial_snapshots(self):
        """Fetch initial orderbook snapshots in background.

        This populates the baseline orderbooks that WebSocket deltas will update.
        Without snapshots, deltas can't be applied correctly.
        """
        snapshot_count = 0
        failed_count = 0

        for ticker, category in self.active_tickers.items():
            try:
                # Fetch REST API snapshot (depth=10 for decent book visibility)
                response = self.api_client.get_orderbook(ticker, depth=10)
                # REST returns {"orderbook": {"yes": [...], "no": [...]}},
                # but WS snapshots/deltas use top-level yes_dollars_fp/no_dollars_fp.
                # Unwrap and normalize to the WS format so _build_observation works.
                book = response.get("orderbook", response)
                # Only store if we don't already have a WS snapshot (WS is fresher)
                if ticker not in self.state.orderbooks:
                    self.state.orderbooks[ticker] = book
                snapshot_count += 1

                # Log progress every 50 markets
                if snapshot_count % 50 == 0:
                    logger.info(f"Snapshot progress: {snapshot_count}/{len(self.active_tickers)}")

                # Small delay to avoid REST API rate limits (10 req/s safe)
                await asyncio.sleep(0.1)

            except Exception as e:
                failed_count += 1
                logger.debug(f"Could not fetch snapshot for {ticker}: {e}")

        logger.info(f"✓ Fetched {snapshot_count}/{len(self.active_tickers)} initial orderbook snapshots ({failed_count} failed)")

    async def _find_all_active_tickers(self, category: str) -> list[str]:
        """Find ALL active (open) market tickers for a category."""
        try:
            # Search for all markets with this series ticker
            response = self.api_client.get_markets(
                series_ticker=category,
                status="open",
                limit=200  # Get up to 200 markets per category
            )

            markets = response.get("markets", [])
            tickers = [m["ticker"] for m in markets]

            return tickers

        except KalshiAPIError as e:
            logger.error(f"Error finding tickers for {category}: {e}")
            return []

    def _create_orderbook_callback(self, category: str, ticker: str):
        """Create orderbook callback for a category."""

        async def callback(ticker_msg: str, orderbook_data: dict):
            """Handle orderbook update (snapshot or delta)."""
            # Check if category is halted
            if category in self.state.halted_categories:
                logger.debug(f"Category {category} is halted")
                return

            # Check risk limits
            if not self._check_risk_limits():
                logger.debug(f"Risk limits prevent trading")
                return

            # Apply delta update to stored orderbook if this is a delta message
            # Delta messages have: price_dollars, delta_fp, side, ts, ts_ms
            # Snapshot messages have: yes, no (or yes_dollars_fp, no_dollars_fp)
            is_delta = "price_dollars" in orderbook_data and "delta_fp" in orderbook_data
            if is_delta:
                # Delta update — apply to stored orderbook
                self._apply_orderbook_delta(ticker, orderbook_data)
            else:
                # Snapshot — store it as the baseline
                self.state.orderbooks[ticker] = orderbook_data

            # Get the complete orderbook (snapshot + applied deltas)
            complete_orderbook = self.state.orderbooks.get(ticker)
            if not complete_orderbook:
                # No baseline yet (REST snapshot still loading) — skip silently
                return

            # Build observation from complete orderbook
            obs = self._build_observation(category, ticker, complete_orderbook)

            if obs is None:
                # One-sided or empty book — skip without spamming logs
                return

            # Get model prediction
            model = self.models[category]
            action, _ = model.predict(obs, deterministic=True)

            # Log raw action and key obs features for debugging
            hs_raw, sk_raw = float(action[0]), float(action[1])
            hs_scaled, sk_scaled = scale_action(action, self.config)
            logger.info(f"Action {category}/{ticker}: mid={obs[0]:.3f} hs={hs_scaled:.3f} sk={sk_scaled:.3f} raw=[{hs_raw:.3f},{sk_raw:.3f}]")

            # Execute action — pass the COMPLETE orderbook, not the raw delta
            await self._execute_action(category, ticker, action, complete_orderbook)

        return callback

    def _apply_orderbook_delta(self, ticker: str, delta: dict):
        """Apply a delta update to the stored orderbook.

        Delta format: {
            'price_dollars': 0.50,  # Price level
            'delta_fp': 100.0,      # Change in quantity (can be negative)
            'side': 'yes' or 'no',  # Which side of the book
            ...
        }
        """
        if ticker not in self.state.orderbooks:
            # No baseline orderbook - can't apply delta without snapshot
            logger.debug(f"Cannot apply delta for {ticker} - no baseline orderbook")
            return

        orderbook = self.state.orderbooks[ticker]
        price = float(delta.get("price_dollars", 0))
        delta_qty = float(delta.get("delta_fp", 0))
        side = delta.get("side", "")

        # Determine which side to update
        # "yes" side = bids (people buying YES contracts)
        # "no" side = asks (people selling YES contracts, i.e., buying NO)
        if side == "yes":
            levels_key = "yes_dollars_fp"
            if levels_key not in orderbook:
                levels_key = "yes_dollars"
            if levels_key not in orderbook:
                levels_key = "yes"
        elif side == "no":
            levels_key = "no_dollars_fp"
            if levels_key not in orderbook:
                levels_key = "no_dollars"
            if levels_key not in orderbook:
                levels_key = "no"
        else:
            logger.debug(f"Unknown side '{side}' in delta for {ticker}")
            return

        if levels_key not in orderbook:
            orderbook[levels_key] = []

        levels = orderbook[levels_key]

        # Find and update the price level
        found = False
        for i, level in enumerate(levels):
            level_price = float(level[0])
            if abs(level_price - price) < 0.0001:  # Match with tolerance
                # Update quantity
                current_qty = float(level[1])
                new_qty = current_qty + delta_qty

                if new_qty <= 0:
                    # Remove level
                    levels.pop(i)
                else:
                    # Update level
                    levels[i] = [str(price), str(new_qty)]

                found = True
                break

        # If level not found and delta is positive, add new level
        if not found and delta_qty > 0:
            levels.append([str(price), str(delta_qty)])

            # Sort levels (bids descending, asks ascending by price)
            # For YES (bids): highest price first
            # For NO (asks): lowest price first (but remember NO prices are inverted for YES asks)
            if side == "yes":
                levels.sort(key=lambda x: -float(x[0]))  # Descending
            else:
                levels.sort(key=lambda x: float(x[0]))   # Ascending

    def _build_observation(self, category: str, ticker: str, orderbook_data: dict) -> Optional[np.ndarray]:
        """Build 16-dimensional observation vector matching mm_env.py exactly.

        All normalization constants must match training (mm_env.py) or the model
        receives out-of-distribution inputs and outputs garbage actions.
        """
        try:
            # Extract orderbook from WebSocket/REST format: [["0.0100", "305.00"], ...]
            # yes_dollars = bids for YES contracts (prices people will pay for YES)
            # no_dollars = bids for NO contracts (YES ask = 1 - no_price)
            bids_raw = orderbook_data.get("yes_dollars_fp", orderbook_data.get("yes_dollars", orderbook_data.get("yes", [])))
            asks_raw = orderbook_data.get("no_dollars_fp", orderbook_data.get("no_dollars", orderbook_data.get("no", [])))

            # One-sided or empty book — skip silently (very common for illiquid markets)
            if not bids_raw or not asks_raw:
                return None

            # YES bid = yes_dollars price directly (highest first)
            best_bid = float(bids_raw[0][0]) if bids_raw else 0.01
            # YES ask = 1 - no_dollars price (NO bids are inverted to get YES asks)
            best_ask = 1.0 - float(asks_raw[0][0]) if asks_raw else 0.99

            # Sanity: ensure ask > bid (crossed market)
            if best_ask <= best_bid:
                return None

            # Skip illiquid markets — if raw spread > 1.0, skip (testing threshold raised)
            # NOTE: Model trained on spreads < 0.30, current live spreads are 0.54-0.99
            # This threshold allows trading but model may see out-of-distribution inputs
            raw_spread = best_ask - best_bid
            if raw_spread > 1.0:
                logger.debug(f"{ticker}: Spread too wide ({raw_spread:.3f}), skipping")
                return None

            # [0] mid_price — same as mm_env.py
            mid_price = (best_bid + best_ask) / 2.0

            # Track mid history for momentum (per-ticker)
            self.state.mid_history[ticker].append(mid_price)
            # Keep last 10 entries to avoid unbounded growth
            if len(self.state.mid_history[ticker]) > 10:
                self.state.mid_history[ticker] = self.state.mid_history[ticker][-10:]

            # [1] spread — clipped same as mm_env.py
            spread = np.clip(best_ask - best_bid, 0.01, 0.10)

            # [2-7] orderbook depths — mm_env.py normalizes by /100.0, NOT /1000.0
            # YES bid depths come directly from yes_dollars orderbook
            bid_l0 = min(float(bids_raw[0][1]) / 100.0, 1.0) if len(bids_raw) > 0 else 0.1
            # YES ask depths come from NO orderbook (same sizes, inverted prices)
            ask_l0 = min(float(asks_raw[0][1]) / 100.0, 1.0) if len(asks_raw) > 0 else 0.1
            bid_l1 = min(float(bids_raw[1][1]) / 100.0, 1.0) if len(bids_raw) > 1 else 0.05
            ask_l1 = min(float(asks_raw[1][1]) / 100.0, 1.0) if len(asks_raw) > 1 else 0.05
            bid_l2 = min(float(bids_raw[2][1]) / 100.0, 1.0) if len(bids_raw) > 2 else 0.02
            ask_l2 = min(float(asks_raw[2][1]) / 100.0, 1.0) if len(asks_raw) > 2 else 0.02

            # [8] book_imbalance — same as mm_env.py
            total_bid = sum(float(b[1]) for b in bids_raw[:3])
            total_ask = sum(float(a[1]) for a in asks_raw[:3])
            book_imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)

            # [9] trade_volume_1m — mm_env.py uses len(trades)/50.0
            # Live: approximate from depth since we don't have trade count per window
            trade_volume = min(1.0, (total_bid + total_ask) / 5000.0)

            # [10] inventory_norm — mm_env.py: inventory / max_inventory (20, not 100)
            inventory = self.state.positions.get(ticker, 0)
            max_inv = self.config.max_inventory  # 20 in training
            inv_norm = inventory / max(max_inv, 1)

            # [11] unrealized_pnl_norm — mm_env.py: pnl / (max_inventory * 0.5)
            entry_price = self.state.entry_prices.get(ticker, mid_price)
            unrealized_pnl = inventory * (mid_price - entry_price) if inventory != 0 else 0.0
            unrealized_norm = unrealized_pnl / max(max_inv * 0.5, 1.0)

            # [12] tte_log — mm_env.py: log(1 + tte_hours), starts at 24 and decrements
            close_time = self.state.close_times.get(ticker)
            if close_time:
                # Calculate actual hours until close
                now = datetime.now(close_time.tzinfo)
                tte_hours = max(0.0, (close_time - now).total_seconds() / 3600.0)
            else:
                tte_hours = 24.0  # fallback
            tte_log = float(np.log(1.0 + tte_hours))

            # [13] momentum — mm_env.py: mid - mid_5_steps_ago
            hist = self.state.mid_history[ticker]
            momentum = mid_price - hist[-5] if len(hist) >= 5 else 0.0

            # [14] realized_pnl_norm — mm_env.py: realized_pnl / 50.0
            realized_pnl_norm = self.state.daily_pnl / 50.0

            # [15] fills_ratio — mm_env.py: (fills_buy + fills_sell) / quote_size
            quote_size = max(self.config.quote_size, 1.0)
            fills_ratio = (self.state.fills_buy[ticker] + self.state.fills_sell[ticker]) / quote_size

            # Build observation and clip to match mm_env.py observation_space bounds
            obs = np.array([
                mid_price,       # [0]
                spread,          # [1]
                bid_l0,          # [2]
                ask_l0,          # [3]
                bid_l1,          # [4]
                ask_l1,          # [5]
                bid_l2,          # [6]
                ask_l2,          # [7]
                book_imbalance,  # [8]
                trade_volume,    # [9]
                inv_norm,        # [10]
                unrealized_norm, # [11]
                tte_log,         # [12]
                momentum,        # [13]
                realized_pnl_norm, # [14]
                fills_ratio,     # [15]
            ], dtype=np.float32)

            # Clip to observation_space bounds (same as mm_env.py line 272)
            obs_low = np.array([0.01, 0.01, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -0.1, -1, 0], dtype=np.float32)
            obs_high = np.array([0.99, 0.10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 0.1, 1, 1], dtype=np.float32)
            obs = np.clip(obs, obs_low, obs_high)

            # Detailed observation breakdown — debug level to avoid log bloat
            logger.debug(
                f"OBS {ticker}: bid={best_bid:.4f} ask={best_ask:.4f} mid={mid_price:.4f} "
                f"spread={spread:.4f} depths=[{bid_l0:.3f},{ask_l0:.3f},{bid_l1:.3f},{ask_l1:.3f}] "
                f"imb={book_imbalance:.3f} vol={trade_volume:.3f} inv={inv_norm:.3f} "
                f"tte={tte_log:.3f} mom={momentum:.4f} rpnl={realized_pnl_norm:.3f} fills={fills_ratio:.3f}"
            )

            return obs

        except Exception as e:
            logger.error(f"Error building observation for {category}/{ticker}: {e}")
            return None

    async def _execute_action(self, category: str, ticker: str, action: np.ndarray, orderbook_data: dict):
        """Execute market-making action via Kalshi API.

        The model outputs [half_spread_control, skew_control] in range [-1, 1].
        We convert this to bid/ask quotes and place limit orders.
        """
        try:
            # Get current orderbook
            # WebSocket sends yes_dollars_fp/no_dollars_fp with string values
            yes_orders = orderbook_data.get("yes_dollars_fp", orderbook_data.get("yes", []))
            no_orders = orderbook_data.get("no_dollars_fp", orderbook_data.get("no", []))

            if not yes_orders or not no_orders:
                logger.debug(f"Missing orderbook data for {category}/{ticker}")
                return

            # Parse best bid/ask - WebSocket format: [["0.0100", "305.00"], ...]
            # YES bid = yes_dollars price directly
            best_bid = float(yes_orders[0][0]) if yes_orders else 0.01
            # YES ask = 1 - no_dollars price (NO bids inverted to YES asks)
            best_ask = 1.0 - float(no_orders[0][0]) if no_orders else 0.99

            # Calculate mid price (cast to Python float to avoid numpy float32 noise)
            mid = float((best_bid + best_ask) / 2.0)

            # Convert model action to half_spread and skew
            # scale_action maps [-1,1] to [0.01, 0.10] for half_spread and [-0.05, 0.05] for skew
            half_spread, skew = scale_action(action, self.config)
            half_spread = float(half_spread)
            skew = float(skew)

            # Calculate our bid/ask quotes
            # Formula from mm_env.py: bid = mid - half_spread + skew, ask = mid + half_spread + skew
            our_bid = round(mid - half_spread + skew, 3)
            our_ask = round(mid + half_spread + skew, 3)

            # Clamp to valid Kalshi range [0.01, 0.99]
            our_bid = max(0.01, min(0.99, our_bid))
            our_ask = max(0.01, min(0.99, our_ask))

            # Ensure ask > bid (prevent crossing)
            if our_ask <= our_bid:
                logger.debug(f"Skipping {ticker}: quotes would cross (bid={our_bid:.3f}, ask={our_ask:.3f})")
                return

            # Check live tick size for this market
            tick_size = self._tick_sizes.get(ticker, 0.01)
            supports_subpenny = tick_size <= 0.001

            if supports_subpenny and self.config.subpenny_enabled:
                # Apply subpenny adjustment for queue priority (+0.001 bid, -0.001 ask)
                our_bid = round(our_bid + 0.001, 3)
                our_ask = round(our_ask - 0.001, 3)
                # Re-check crossing after adjustment
                if our_ask <= our_bid:
                    logger.debug(f"Skipping {ticker}: quotes cross after subpenny adjustment")
                    return
                our_bid_cents = round(our_bid * 100, 1)
                our_ask_cents = round(our_ask * 100, 1)
            else:
                # Whole-cent pricing
                our_bid = round(our_bid, 2)
                our_ask = round(our_ask, 2)
                our_bid_cents = int(our_bid * 100)
                our_ask_cents = int(our_ask * 100)

            # Check inventory limits
            cat_config = next(c for c in JUNE_TOP5_CATEGORIES if c.name == category)
            current_inventory = self.state.positions.get(ticker, 0)

            # Place bid (buy) if within inventory limits
            if abs(current_inventory + 1) <= cat_config.max_inventory:
                await self._send_order(
                    category=category,
                    ticker=ticker,
                    side="buy",
                    price=our_bid_cents,
                    size=1
                )
                self.state.quotes_sent += 1

            # Place ask (sell) if within inventory limits
            if abs(current_inventory - 1) <= cat_config.max_inventory:
                await self._send_order(
                    category=category,
                    ticker=ticker,
                    side="sell",
                    price=our_ask_cents,
                    size=1
                )
                self.state.quotes_sent += 1

        except Exception as e:
            logger.error(f"Error executing action for {category}/{ticker}: {e}")

    async def _send_order(self, category: str, ticker: str, side: Literal["buy", "sell"],
                          price: float, size: int):
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

                        elif status == "resting":
                            # Cancel stale unfilled orders after 30 seconds
                            # Frees up capital for new quotes
                            age = (datetime.now() - order_info["timestamp"]).total_seconds()
                            if age > 30:
                                try:
                                    self.api_client.cancel_order(order_id)
                                    logger.info(f"Canceled stale order {order_id} ({order_info['ticker']} {order_info['side']}, age={age:.0f}s)")
                                    del self.state.pending_orders[order_id]
                                except Exception as e:
                                    logger.debug(f"Could not cancel stale order {order_id}: {e}")

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

        # Update position and track entry price (matching mm_env.py _fill_buy/_fill_sell)
        price_dollars = price / 100.0
        old_inv = self.state.positions[ticker]
        if side == "buy":
            self.state.positions[ticker] += size
            self.state.fills_buy[ticker] += size
            # Update weighted avg entry price
            new_inv = self.state.positions[ticker]
            if new_inv > 0:
                old_entry = self.state.entry_prices.get(ticker, price_dollars)
                self.state.entry_prices[ticker] = (old_inv * old_entry + size * price_dollars) / new_inv
        else:  # sell
            self.state.positions[ticker] -= size
            self.state.fills_sell[ticker] += size
            new_inv = self.state.positions[ticker]
            # Reset entry price on direction flip
            if new_inv != 0 and ((old_inv > 0 and new_inv < 0) or (old_inv < 0 and new_inv > 0)):
                self.state.entry_prices[ticker] = price_dollars
            elif new_inv == 0:
                self.state.entry_prices.pop(ticker, None)

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
            asyncio.create_task(self._check_fills_loop()),
            asyncio.create_task(self._daily_reset_loop()),
            asyncio.create_task(self._hourly_report_loop()),
        ]

        # Add WebSocket manager with auto-reconnection, or use REST polling
        if self.ws_client:
            tasks.append(asyncio.create_task(self._websocket_manager_loop()))
        else:
            # No WebSocket - use REST API polling for market data
            tasks.append(asyncio.create_task(self._rest_polling_loop()))

        # Wait for tasks
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _rest_polling_loop(self):
        """Poll market data via REST API when WebSocket is unavailable."""
        logger.info("Starting REST API polling loop (checking markets every 10 seconds)")

        while self.running:
            await asyncio.sleep(10)  # Poll every 10 seconds

            # Check risk limits
            if not self._check_risk_limits():
                continue

            # Iterate through all active markets
            for ticker, category in self.active_tickers.items():
                # Skip halted categories
                if category in self.state.halted_categories:
                    continue

                try:
                    # Fetch orderbook from REST API
                    response = self.api_client.get_orderbook(ticker)
                    orderbook = response.get("orderbook", {})

                    if not orderbook:
                        continue

                    # Build observation from orderbook
                    obs = self._build_observation(category, ticker, orderbook)

                    if obs is None:
                        continue

                    # Get model prediction
                    model = self.models[category]
                    action, _ = model.predict(obs, deterministic=True)

                    # Execute action
                    await self._execute_action(category, ticker, action, orderbook)

                except KalshiAPIError as e:
                    logger.error(f"Error polling {category}/{ticker}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in REST polling for {category}/{ticker}: {e}")

    async def _websocket_manager_loop(self):
        """Manage WebSocket connection with auto-reconnection."""
        reconnect_delay = 5  # seconds

        while self.running:
            try:
                logger.info("Connecting to WebSocket...")
                await self.ws_client.connect()

                # Re-subscribe to all markets after reconnection
                logger.info(f"Re-subscribing to {len(self.active_tickers)} markets...")
                for ticker, category in self.active_tickers.items():
                    await self.ws_client.subscribe_orderbook(
                        ticker,
                        self._create_orderbook_callback(category, ticker)
                    )

                logger.info("WebSocket connected and subscribed, starting listener...")
                # Listen until connection closes
                await self.ws_client.listen()

                # Connection closed
                logger.warning("WebSocket disconnected, reconnecting in 5 seconds...")
                await asyncio.sleep(reconnect_delay)

            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)

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
                 base_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.ws = None
        self.subscriptions: Dict[str, callable] = {}  # ticker -> callback

        # Load RSA private key for signing
        try:
            with open(api_secret, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        except Exception as e:
            logger.warning(f"Could not load private key from {api_secret}: {e}")
            self.private_key = None

    async def connect(self):
        """Connect to Kalshi WebSocket with proper RSA-PSS authentication."""
        logger.info(f"Connecting to WebSocket: {self.base_url}")

        if not self.private_key:
            logger.error("No private key loaded - cannot authenticate WebSocket")
            return False

        try:
            # Generate timestamp in milliseconds
            timestamp = str(int(time.time() * 1000))

            # Create message to sign: timestamp + method + path
            message = f"{timestamp}GET/trade-api/ws/v2"

            # Sign with RSA-PSS
            signature_bytes = self.private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH
                ),
                hashes.SHA256()
            )
            signature = base64.b64encode(signature_bytes).decode('utf-8')

            # Connect with authentication headers
            self.ws = await websockets.connect(
                self.base_url,
                additional_headers={
                    "KALSHI-ACCESS-KEY": self.api_key,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                    "KALSHI-ACCESS-TIMESTAMP": timestamp,
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
        logger.info("WebSocket listener started - waiting for messages...")
        try:
            async for message in self.ws:
                logger.debug(f"WebSocket message received: {message[:200]}...")  # Log first 200 chars
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

        logger.debug(f"WebSocket message type: {msg_type}, ticker: {ticker}")

        # Handle both orderbook_snapshot (initial state) and orderbook_delta (updates)
        if msg_type in ["orderbook_snapshot", "orderbook_delta"] and ticker in self.subscriptions:
            logger.info(f"Processing {msg_type} for {ticker}")
            callback = self.subscriptions[ticker]
            await callback(ticker, data["msg"])
        else:
            logger.debug(f"Ignoring message type '{msg_type}' for ticker '{ticker}' (not subscribed or different type)")


async def main():
    """Main entry point."""
    # Check for paper mode flag
    paper_mode = os.getenv("PAPER_MODE", "false").lower() == "true"

    logger.info("=" * 80)
    logger.info("KALSHI LIVE TRADING BOT - PHASE 1")
    logger.info("=" * 80)
    logger.info(f"Mode: {'PAPER TRADING' if paper_mode else 'LIVE TRADING'}")
    logger.info(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    logger.info(f"Categories: {len(JUNE_TOP5_CATEGORIES)}")
    logger.info(f"Risk Limits: Daily loss ${TRADING_CONFIG['max_daily_loss']}, Stop loss ${TRADING_CONFIG['stop_loss_threshold']}")
    logger.info("=" * 80)

    trader = LiveTrader(paper_mode=paper_mode)
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
