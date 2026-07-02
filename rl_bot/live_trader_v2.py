"""
Live Trading Bot - June 2026 Deployment
Active categories based on June 29-30 trading data
Capital: $95.37 / 5 = $19.07 per category
Selection: Top 5 by June trade volume with trained models
"""

import argparse
import asyncio
import base64
import fcntl
import importlib
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
from rl_bot.live_book import (LiveBook, clamp_quotes, quote_edge_ok, sample_mid,
                              trade_window_features)
from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import scale_action
from rl_bot.mm_metadata import MarketMetadataLoader
from rl_bot.reward import compute_maker_fee, compute_taker_fee


# --- Dynamic config loading (Step 1) ---
# Maps short names to module paths; also accepts a file path directly.
_CONFIG_ALIASES = {
    "lowvol": "rl_bot.live_config_lowvol",
    "june": "rl_bot.live_config_june",
    "conservative": "rl_bot.live_config_conservative",
    "full": "rl_bot.live_config_full",
    "top50": "rl_bot.live_config_top50",
    "tier1": "rl_bot.live_config_tier1_tested",
    "262": "rl_bot.live_config_262_tested",
    "minimal": "rl_bot.live_config_minimal",
    "btc_demo": "rl_bot.live_config_btc_demo",
}


def _load_config(name: str):
    """Load CATEGORIES, TRADING_CONFIG, MONITORING_CONFIG from a config module.

    Args:
        name: alias (e.g. "june") or dotted module path or file path
    Returns:
        (categories_list, trading_config_dict, monitoring_config_dict)
    """
    module_path = _CONFIG_ALIASES.get(name, name)

    # If it looks like a file path, convert to module path
    if module_path.endswith(".py"):
        module_path = module_path.replace("/", ".").replace("\\", ".").removesuffix(".py")

    mod = importlib.import_module(module_path)

    # Each config module exports a top-level list + two dicts.
    # Naming varies: CATEGORIES, CONSERVATIVE_CATEGORIES, etc.
    # Find the first list of CategoryConfig objects.
    categories = None
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if isinstance(obj, list) and len(obj) > 0 and hasattr(obj[0], "name") and hasattr(obj[0], "max_inventory"):
            categories = obj
            break

    if categories is None:
        raise ValueError(f"Config module {module_path} has no CategoryConfig list")

    trading_cfg = getattr(mod, "TRADING_CONFIG", {})
    monitoring_cfg = getattr(mod, "MONITORING_CONFIG", {})
    return categories, trading_cfg, monitoring_cfg


# Parse CLI args early so they're available at module level
def _parse_args():
    parser = argparse.ArgumentParser(description="Kalshi Live Trading Bot")
    # Default is the Phase 2 low-vol whitelist — the only config whose
    # categories have demonstrated live profitability. Others opt-in via flag.
    parser.add_argument("--config", default=os.getenv("TRADING_CONFIG", "lowvol"),
                        help="Config name (lowvol/june/conservative/full/...) or module path")
    return parser.parse_known_args()[0]

_cli_args = _parse_args()
CATEGORIES, TRADING_CONFIG, MONITORING_CONFIG = _load_config(_cli_args.config)

# Load environment variables
load_dotenv()

# Configure logging — one log file per config so concurrent runs with different
# configs never interleave in the same file (was hardcoded 'live_trading.log')
_log_name = _cli_args.config.replace("/", "_").replace("\\", "_").removesuffix(".py")
logging.basicConfig(
    level=logging.INFO,  # Back to INFO - DEBUG was too verbose
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'live_trading_{_log_name}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def acquire_single_instance_lock() -> Optional[object]:
    """Take an exclusive non-blocking flock so only ONE live trader can run
    per account. Two processes quoting the same account cancel each other's
    orders and double-place quotes (observed in the 6/30 logs as duplicate
    ORDER bursts). Returns the open file handle (must stay referenced for the
    process lifetime — the lock releases when it closes) or None if another
    instance already holds the lock.
    """
    lock_path = Path.home() / ".kalshi_mm_live.lock"
    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def realized_vol(mid_history: list) -> float:
    """Rolling stddev of the last 20 mid prices — same formula as mm_env obs[19].

    Returns 0.0 with fewer than 3 samples (not enough signal to estimate vol).
    """
    if len(mid_history) < 3:
        return 0.0
    return float(np.std(mid_history[-20:]))


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
        self.orderbooks: Dict[str, LiveBook] = {}  # ticker -> canonical live book (REST + WS merged)
        # Per-ticker quote throttle (Step 5): skip if last quote < 1s ago
        self.last_quote_time: Dict[str, float] = {}  # ticker -> monotonic timestamp
        # Per-ticker active resting orders (Step 6): cancel-before-replace
        self.active_orders: Dict[str, Dict[str, Optional[str]]] = {}  # ticker -> {"bid_id": ..., "ask_id": ...}
        # Last quoted (bid, ask) per ticker — re-quote only when the desired
        # price moves >= 1 tick, otherwise keep the resting orders and their
        # queue priority (constant cancel-replace = permanent back of queue)
        self.quoted_prices: Dict[str, tuple] = {}  # ticker -> (bid_dollars, ask_dollars)
        # Fill reconciliation bookkeeping (fills come from GET /portfolio/fills)
        self.processed_fill_ids: set = set()
        # Quoting pause deadline after an insufficient_balance rejection
        self.balance_backoff_until: float = 0.0
        # Taker/maker split + total fees — the Phase 2 deployment gate is
        # "taker fraction < 10% and fees match the maker schedule"
        self.taker_fills: int = 0
        self.maker_fills: int = 0
        self.fees_paid: float = 0.0
        # Mid history sample clock per ticker (60s cadence, matching training)
        self.last_mid_sample: Dict[str, float] = defaultdict(float)
        # Recent trade prints per ticker from the WS "trade" channel:
        # (epoch_s, contracts, taker_side). Feeds obs [9] volume and [16] flow
        # with REAL trade data instead of the old book-depth proxies.
        self.recent_trades: Dict[str, list] = defaultdict(list)
        # Per-ticker realized PnL — obs [14] in training is per-episode
        # (per-ticker), not the account-wide daily total
        self.realized_by_ticker: Dict[str, float] = defaultdict(float)

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate."""
        return self.fills / self.quotes_sent if self.quotes_sent > 0 else 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def position_value(self, active_tickers: set | None = None) -> float:
        """Capital locked in open positions (contracts * entry price).

        With `active_tickers`, counts only markets this bot quotes. Legacy
        positions from prior deployments (117 of them, unclosable or frozen
        to settlement) otherwise blow through max_position_value at the 0.50
        fallback entry and block ALL new quoting — the limit exists to cap
        risk THIS bot adds, not inventory it can't do anything about.
        """
        total = 0.0
        for ticker, inv in self.positions.items():
            if inv == 0:
                continue
            if active_tickers is not None and ticker not in active_tickers:
                continue
            entry = self.entry_prices.get(ticker, 0.50)  # fallback to 0.50 if unknown
            total += abs(inv) * entry
        return total

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
        """Calculate unrealized PnL for a ticker using actual entry price."""
        position = self.positions[ticker]
        if position == 0:
            return 0.0
        entry = self.entry_prices.get(ticker)
        if entry is None:
            return 0.0
        # last_price and entry are both in dollars [0.01, 0.99]
        return position * (last_price - entry)

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
        logger.info(f"Config: {_cli_args.config}")
        logger.info(f"Categories ({len(CATEGORIES)}): {[c.name for c in CATEGORIES]}")
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

                # Get current positions — V2 nests them under "market_positions"
                # with position_fp (the old "positions" key never existed in V2,
                # so this loop silently saw zero positions for months)
                positions_response = self.api_client.get_positions()
                positions = {}
                raw_positions = positions_response.get(
                    "market_positions", positions_response.get("positions", []))
                for pos in raw_positions:
                    ticker = pos.get("ticker") or pos.get("market_ticker")
                    position = int(round(float(pos.get("position_fp", pos.get("position", 0)) or 0)))
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

        for cat in CATEGORIES:
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
                # Guard: the model's observation space must match the live obs
                # builder (20-dim). Old 16/19-dim checkpoints error (or worse,
                # act on garbage) when fed 20-dim observations — the june_*
                # models deployed on 6/30 were 16-dim against a 20-dim builder.
                obs_shape = getattr(model.observation_space, "shape", None)
                if obs_shape != (20,):
                    logger.warning(f"✗ {cat.name}: {checkpoint_path.name} expects obs "
                                   f"{obs_shape}, live builder is (20,) — category "
                                   f"DISABLED until retrained (see hpc/ jobs)")
                    continue
                self.models[cat.name] = model
                logger.info(f"✓ Loaded model: {cat.name} ({checkpoint_path.name})")
            except Exception as e:
                logger.error(f"✗ Failed to load model {cat.name}: {e}")

        logger.info(f"Loaded {len(self.models)} models")

        # Categories without a compatible model are dropped (warned above);
        # refuse to start only if NOTHING is tradeable.
        missing = [c.name for c in CATEGORIES if c.name not in self.models]
        if missing:
            logger.warning(f"Categories without compatible checkpoints (skipped): {missing}")
        if not self.models:
            raise RuntimeError(f"No category has a 20-dim checkpoint in {checkpoints_dir}/ — "
                               f"nothing to trade. Retrain or sync models first.")

        # Load market metadata for subpenny tick size validation
        markets_path = Path("output/rl_all_markets_3mo.parquet")
        if markets_path.exists():
            self.metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=str(markets_path))
            logger.info("✓ Loaded market metadata for subpenny pricing")
        else:
            logger.warning("Market metadata not found — subpenny pricing disabled")

        # Get ALL active tickers for each category
        logger.info("Finding active markets...")
        for cat in CATEGORIES:
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
            self.ws_client.trade_callback = self._on_trade
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
                # REST wraps the book in {"orderbook": {...}}; LiveBook accepts
                # yes/no, yes_dollars, and yes_dollars_fp key variants and only
                # replaces the sides present, so WS one-sided snapshots merge.
                book = self.state.orderbooks.setdefault(ticker, LiveBook())
                book.load_snapshot(response.get("orderbook", response))
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
        """Find ALL active (open) market tickers for a category.

        Validates that returned tickers actually start with the category prefix
        to guard against API returning unrelated markets (prefix aliasing).
        Only returns tickers for categories that have a loaded model.
        """
        # Guard: only discover tickers for categories with loaded models
        if category not in self.models:
            logger.warning(f"Skipping ticker discovery for {category} — no loaded model")
            return []

        try:
            # Search for all markets with this series ticker
            response = self.api_client.get_markets(
                series_ticker=category,
                status="open",
                limit=200  # Get up to 200 markets per category
            )

            markets = response.get("markets", [])
            tickers = []
            for m in markets:
                ticker = m["ticker"]
                # Validate the ticker actually belongs to this category.
                # Kalshi tickers follow {SERIES}-{DATE}-{STRIKE} format,
                # e.g. "KXBTC-26JUL01-T50250". The series prefix must match.
                if ticker.startswith(category):
                    tickers.append(ticker)
                else:
                    logger.warning(
                        f"Filtered out ticker {ticker} — does not match category {category}"
                    )

            return tickers

        except KalshiAPIError as e:
            logger.error(f"Error finding tickers for {category}: {e}")
            return []

    def _create_orderbook_callback(self, category: str, ticker: str):
        """Create orderbook callback for a category."""

        async def callback(ticker_msg: str, orderbook_data: dict):
            """Handle orderbook update (snapshot or delta)."""
            try:
                # Check if category is halted
                if category in self.state.halted_categories:
                    logger.debug(f"Category {category} is halted")
                    return

                # Check risk limits
                if not self._check_risk_limits():
                    logger.debug(f"Risk limits prevent trading")
                    return

                # Route the message into the canonical LiveBook. Delta messages
                # have price_dollars + delta_fp; snapshot messages carry yes/no
                # level arrays (often one side per message — LiveBook merges).
                book = self.state.orderbooks.setdefault(ticker, LiveBook())
                if "price_dollars" in orderbook_data and "delta_fp" in orderbook_data:
                    book.apply_delta(
                        orderbook_data.get("side", ""),
                        float(orderbook_data.get("price_dollars", 0)),
                        float(orderbook_data.get("delta_fp", 0)),
                    )
                else:
                    book.load_snapshot(orderbook_data)

                # One-sided, empty, or crossed book — nothing safe to do
                if not book.is_valid():
                    return

                # Build observation from the canonical book
                obs = self._build_observation(category, ticker, book)

                if obs is None:
                    # One-sided or empty book — skip without spamming logs
                    return

                # Guard: only trade if this category has a loaded model
                if category not in self.models:
                    logger.debug(f"No model for {category}, skipping {ticker}")
                    return

                # Get model prediction
                model = self.models[category]
                action, _ = model.predict(obs, deterministic=True)

                # Raw action + key obs — debug level: this fires on every book
                # tick (45k lines in 6 min at INFO = gigabytes of log per day)
                hs_raw, sk_raw = float(action[0]), float(action[1])
                hs_scaled, sk_scaled = scale_action(action, self.config)
                logger.debug(f"Action {category}/{ticker}: mid={obs[0]:.3f} hs={hs_scaled:.3f} sk={sk_scaled:.3f} raw=[{hs_raw:.3f},{sk_raw:.3f}]")

                # Execute action against the canonical book
                await self._execute_action(category, ticker, action, book)

            except Exception as e:
                logger.error(f"Callback error {category}/{ticker}: {e}", exc_info=True)

        return callback

    async def _on_trade(self, ticker: str, msg: dict):
        """Record a public trade print from the WS "trade" channel.

        Stored as (arrival_epoch_s, contracts, taker_side) — the inputs
        trade_window_features() needs to replicate training's obs [9]/[16].
        Arrival time is used rather than the message ts: prints stream in
        real time and it avoids epoch-unit surprises (Kalshi mixes s/ms).
        """
        try:
            count = int(float(msg.get("count_fp", msg.get("count", 1)) or 1))
            taker_side = msg.get("taker_side", "")
            if taker_side in ("yes", "no"):
                self.state.recent_trades[ticker].append((time.time(), count, taker_side))
        except Exception as e:
            logger.debug(f"Bad trade message for {ticker}: {e}")

    def _build_observation(self, category: str, ticker: str, book: LiveBook) -> Optional[np.ndarray]:
        """Build 20-dimensional observation vector matching mm_env.py exactly.

        All normalization constants must match training (mm_env.py) or the model
        receives out-of-distribution inputs and outputs garbage actions.
        """
        try:
            # LiveBook already normalized REST/WS payloads and validated that
            # both sides exist and the book is not crossed (is_valid() checked
            # by the caller). Best levels are computed, never index-dependent.
            if not book.is_valid():
                return None
            best_bid = book.best_bid()
            best_ask = book.best_ask()

            # Skip markets with spread > training clip bound (0.50).
            # Model was trained with spread clipped to [0.01, 0.50]; wider spreads
            # are outside the training distribution and produce garbage actions.
            raw_spread = best_ask - best_bid
            if raw_spread > 0.50:
                logger.warning(f"{ticker}: Spread {raw_spread:.3f} > 0.50 training bound, skipping")
                return None

            # [0] mid_price — same as mm_env.py
            mid_price = (best_bid + best_ask) / 2.0

            # Track mid history on the TRAINING cadence (one entry per 60s
            # window, last entry refreshed in place). Momentum/velocity/vol
            # features and the 0.05 vol threshold were all calibrated on 60s
            # windows — per-tick appends ran them ~60x too fast.
            self.state.last_mid_sample[ticker] = sample_mid(
                self.state.mid_history[ticker],
                self.state.last_mid_sample[ticker],
                time.monotonic(),
                mid_price,
            )

            # Volatility filter: MM loses when prices trend (high vol = informed
            # flow = adverse selection). Skip quoting entirely above the threshold
            # that divides MM-friendly from MM-hostile markets (0.05).
            vol = realized_vol(self.state.mid_history[ticker])
            if vol > self.config.vol_filter_threshold:
                logger.info(f"{ticker}: vol {vol:.4f} > {self.config.vol_filter_threshold} threshold — skipping quote")
                return None

            # [1] spread — clipped same as mm_env.py
            spread = np.clip(best_ask - best_bid, 0.01, 0.50)

            # [2-7] orderbook depths — mm_env.py normalizes by /100.0, NOT /1000.0
            # Levels come best-first from LiveBook (bids desc, asks asc in YES terms)
            bids = book.bid_levels(3)
            asks = book.ask_levels(3)
            bid_l0 = min(bids[0][1] / 100.0, 1.0) if len(bids) > 0 else 0.1
            ask_l0 = min(asks[0][1] / 100.0, 1.0) if len(asks) > 0 else 0.1
            bid_l1 = min(bids[1][1] / 100.0, 1.0) if len(bids) > 1 else 0.05
            ask_l1 = min(asks[1][1] / 100.0, 1.0) if len(asks) > 1 else 0.05
            bid_l2 = min(bids[2][1] / 100.0, 1.0) if len(bids) > 2 else 0.02
            ask_l2 = min(asks[2][1] / 100.0, 1.0) if len(asks) > 2 else 0.02

            # [8] book_imbalance — same as mm_env.py (top-3 depth each side)
            total_bid = sum(q for _, q in bids)
            total_ask = sum(q for _, q in asks)
            book_imbalance = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)

            # [9] trade_volume_1m and [16] flow from REAL trade prints (WS
            # "trade" channel), matching training's per-window trade counts.
            # The old depth proxy measured liquidity, not activity.
            trade_volume, trade_flow = trade_window_features(
                self.state.recent_trades[ticker], time.time())

            # [10] inventory_norm — mm_env.py: inventory / max_inventory (20, not 100)
            inventory = self.state.positions.get(ticker, 0)
            max_inv = self.config.max_inventory  # 20 in training
            inv_norm = inventory / max(max_inv, 1)

            # [11] unrealized_pnl_norm — mm_env.py: pnl / (max_inventory * 0.5)
            entry_price = self.state.entry_prices.get(ticker, mid_price)
            unrealized_pnl = inventory * (mid_price - entry_price) if inventory != 0 else 0.0
            unrealized_norm = unrealized_pnl / max(max_inv * 0.5, 1.0)

            # [12] tte_log — mm_env.py: log(1 + tte_hours). Training episodes
            # START at 24h, so cap live tte there: far-dated markets otherwise
            # produce tte_log up to 4.0 while the model never saw > log(25)=3.22
            close_time = self.state.close_times.get(ticker)
            if close_time:
                # Calculate actual hours until close, capped at training range
                now = datetime.now(close_time.tzinfo)
                tte_hours = min(max(0.0, (close_time - now).total_seconds() / 3600.0), 24.0)
            else:
                tte_hours = 24.0  # fallback
            tte_log = float(np.log(1.0 + tte_hours))

            # [13] momentum — mm_env.py: mid - mid_5_steps_ago
            hist = self.state.mid_history[ticker]
            momentum = mid_price - hist[-5] if len(hist) >= 5 else 0.0

            # [14] realized_pnl_norm — mm_env.py: realized_pnl / 50.0.
            # Per-TICKER, like a training episode — the account-wide daily
            # total previously leaked every other market's PnL into this obs
            realized_pnl_norm = self.state.realized_by_ticker[ticker] / 50.0

            # [15] fills_ratio — mm_env.py: (fills_buy + fills_sell) / quote_size
            quote_size = max(self.config.quote_size, 1.0)
            fills_ratio = (self.state.fills_buy[ticker] + self.state.fills_sell[ticker]) / quote_size

            # --- Anti-adverse-selection features ---

            # [16] Trade flow imbalance — real taker-side flow from the trade
            # channel (computed above with [9]). The old code duplicated the
            # book imbalance here, giving the model the same number twice.
            flow_imbalance = trade_flow

            # [17] Price velocity: rate of mid change over last 3 observations
            if len(hist) >= 3:
                velocity = (mid_price - hist[-3]) / 3.0
            else:
                velocity = 0.0

            # [18] Fill toxicity: imbalance of our own fills for this ticker
            # If we're only getting filled on one side, informed flow is hitting us
            fb = self.state.fills_buy[ticker]
            fs = self.state.fills_sell[ticker]
            total_fills = fb + fs
            if total_fills > 0:
                fill_toxicity = (fb - fs) / total_fills
            else:
                fill_toxicity = 0.0

            # [19] Realized volatility — mm_env.py: std(mid_history[-20:]) / 0.05, clipped
            vol_norm = float(np.clip(vol / 0.05, 0.0, 1.0))

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
                flow_imbalance,  # [16]
                velocity,        # [17]
                fill_toxicity,   # [18]
                vol_norm,        # [19]
            ], dtype=np.float32)

            # Clip to observation_space bounds (same as mm_env.py)
            obs_low = np.array([0.01, 0.01, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -0.1, -1, 0, -1, -0.1, -1, 0], dtype=np.float32)
            obs_high = np.array([0.99, 0.50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 0.1, 1, 1, 1, 0.1, 1, 1], dtype=np.float32)
            obs = np.clip(obs, obs_low, obs_high)

            # Detailed observation breakdown — debug level to avoid log bloat
            logger.debug(
                f"OBS {ticker}: bid={best_bid:.4f} ask={best_ask:.4f} mid={mid_price:.4f} "
                f"spread={spread:.4f} depths=[{bid_l0:.3f},{ask_l0:.3f},{bid_l1:.3f},{ask_l1:.3f}] "
                f"imb={book_imbalance:.3f} vol={trade_volume:.3f} inv={inv_norm:.3f} "
                f"tte={tte_log:.3f} mom={momentum:.4f} rpnl={realized_pnl_norm:.3f} fills={fills_ratio:.3f} "
                f"flow={flow_imbalance:.3f} vel={velocity:.4f} tox={fill_toxicity:.3f} vol={vol_norm:.3f}"
            )

            return obs

        except Exception as e:
            logger.error(f"Error building observation for {category}/{ticker}: {e}")
            return None

    async def _execute_action(self, category: str, ticker: str, action: np.ndarray, book: LiveBook):
        """Execute market-making action via Kalshi API.

        The model outputs [half_spread_control, skew_control] in range [-1, 1].
        We convert this to bid/ask quotes and place PASSIVE (post-only) limit
        orders. Safety layers, in order:
          1. clamp_quotes  — quotes never cross the touch (no accidental taker)
          2. quote_edge_ok — don't quote spreads that ceil'd fees would consume
          3. tick-move check — keep resting orders (and queue priority) unless
             the desired price moved >= 1 tick
          4. post_only=True — the exchange rejects any residual crossing order
        """
        try:
            # Step 5: per-ticker quote throttle — skip if last quote < 1s ago
            now_mono = time.monotonic()
            last_qt = self.state.last_quote_time.get(ticker, 0.0)
            if now_mono - last_qt < 1.0:
                return  # too soon, skip
            self.state.last_quote_time[ticker] = now_mono

            # Capital guard 1: pause all quoting after an insufficient_balance
            # rejection — retrying just burns API rate limit until capital frees
            if time.monotonic() < self.state.balance_backoff_until:
                return

            if not book.is_valid():
                return
            best_bid = book.best_bid()
            best_ask = book.best_ask()
            mid = float((best_bid + best_ask) / 2.0)

            # Capital guard 2: skip extreme-priced books. A 97c contract locks
            # 97c of collateral for at most pennies of spread — near-settlement
            # lottery tickets, not market making.
            if not (self.config.quote_band_lo <= mid <= self.config.quote_band_hi):
                return

            # Capital guard 3: global cap on concurrent resting orders. A
            # two-sided quote locks ~$1; quoting every subscribed market
            # exhausts the balance and every further order is rejected.
            open_orders = sum(1 for pair in self.state.active_orders.values()
                              for oid in pair.values() if oid)
            if open_orders >= self.config.max_open_orders and ticker not in self.state.active_orders:
                return  # budget spent — existing quoted tickers keep refreshing

            # Convert model action to half_spread and skew
            # scale_action maps [-1,1] to [0.01, 0.50] for half_spread and [-0.05, 0.05] for skew
            half_spread, skew = scale_action(action, self.config)
            half_spread = float(half_spread)
            skew = float(skew)

            # Quote formula from mm_env.py: bid/ask = mid -/+ half_spread + skew
            our_bid = mid - half_spread + skew
            our_ask = mid + half_spread + skew

            # Tick size decides price granularity (0.001 = subpenny market)
            tick = self._tick_sizes.get(ticker, 0.01)
            supports_subpenny = tick <= 0.001

            if supports_subpenny and self.config.subpenny_enabled:
                # Subpenny adjustment for queue priority (+0.001 bid, -0.001 ask)
                our_bid += 0.001
                our_ask -= 0.001

            # Snap to the market's tick grid and legal price range
            decimals = 3 if supports_subpenny else 2
            our_bid = max(0.01, min(0.99, round(our_bid, decimals)))
            our_ask = max(0.01, min(0.99, round(our_ask, decimals)))
            if our_ask <= our_bid:
                return  # model quotes collapsed — nothing to do

            # Safety 1: never cross the touch — a crossing "quote" executes as
            # taker (pays the spread + 7% fee), the opposite of market making
            clamped = clamp_quotes(our_bid, our_ask, best_bid, best_ask, tick)
            if clamped is None:
                return
            our_bid, our_ask = clamped

            # Safety 2: fee gate — Kalshi rounds each fill's fee UP to the next
            # cent, so at size=1 a round trip near 0.50 costs $0.02. Quoting a
            # spread that fees consume guarantees losses; skip instead.
            if not quote_edge_ok(our_bid, our_ask, self.config.maker_fee_rate,
                                 size=1, min_edge=self.config.min_quote_edge):
                logger.debug(f"{ticker}: spread {our_ask - our_bid:.3f} fails fee gate, not quoting")
                return

            # Safety 3: keep resting orders when the desired quote hasn't moved
            # a full tick — cancel-replace on every book tick means permanently
            # sitting at the back of the FIFO queue and never earning fills
            prev = self.state.quoted_prices.get(ticker)
            existing = self.state.active_orders.get(ticker, {})
            if (prev is not None
                    and existing.get("bid_id") and existing.get("ask_id")
                    and abs(prev[0] - our_bid) < tick * 0.999
                    and abs(prev[1] - our_ask) < tick * 0.999):
                return  # same quotes still resting — preserve queue priority

            # Cents for the API (1 decimal for subpenny, int for whole-cent)
            if supports_subpenny:
                our_bid_cents = round(our_bid * 100, 1)
                our_ask_cents = round(our_ask * 100, 1)
            else:
                our_bid_cents = int(round(our_bid * 100))
                our_ask_cents = int(round(our_ask * 100))

            # Check inventory limits
            cat_config = next(c for c in CATEGORIES if c.name == category)
            current_inventory = self.state.positions.get(ticker, 0)

            # Step 6: cancel existing resting orders for this ticker before placing new ones
            for side_key in ("bid_id", "ask_id"):
                old_id = existing.get(side_key)
                if old_id:
                    try:
                        self.api_client.cancel_order(old_id)
                        # Remove from pending_orders tracking too
                        self.state.pending_orders.pop(old_id, None)
                    except Exception:
                        pass  # already filled or canceled

            new_bid_id = None
            new_ask_id = None

            # Place bid (buy) if within inventory limits — post-only so the
            # exchange rejects it rather than letting it execute as taker
            if abs(current_inventory + 1) <= cat_config.max_inventory:
                new_bid_id = await self._send_order(
                    category=category,
                    ticker=ticker,
                    side="buy",
                    price=our_bid_cents,
                    size=1,
                    post_only=True,
                )
                self.state.quotes_sent += 1

            # Place ask (sell) if within inventory limits
            if abs(current_inventory - 1) <= cat_config.max_inventory:
                new_ask_id = await self._send_order(
                    category=category,
                    ticker=ticker,
                    side="sell",
                    price=our_ask_cents,
                    size=1,
                    post_only=True,
                )
                self.state.quotes_sent += 1

            # Update active orders map and remember what we quoted
            self.state.active_orders[ticker] = {"bid_id": new_bid_id, "ask_id": new_ask_id}
            self.state.quoted_prices[ticker] = (our_bid, our_ask)

        except Exception as e:
            logger.error(f"Error executing action for {category}/{ticker}: {e}")

    async def _send_order(self, category: str, ticker: str, side: Literal["buy", "sell"],
                          price: float, size: int, post_only: bool = False,
                          time_in_force: str = "good_till_canceled") -> Optional[str]:
        """Send order to Kalshi via REST API. Returns order_id or None.

        post_only=True for quotes (maker-only, exchange rejects crossing);
        time_in_force="immediate_or_cancel" for risk exits that must cross.
        """
        try:
            logger.info(f"ORDER: {category}/{ticker} {side} {size} @ {price}c"
                        f"{' post_only' if post_only else ''}{' IOC' if time_in_force == 'immediate_or_cancel' else ''}")

            # Place order via Kalshi API
            response = self.api_client.place_limit_order(
                ticker=ticker,
                side=side,
                price_cents=price,
                size=size,
                post_only=post_only,
                time_in_force=time_in_force,
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
                logger.info(f"Order placed: {order_id}")
            return order_id

        except KalshiAPIError as e:
            # Out of collateral: stop quoting for a while instead of spamming
            # thousands of doomed order attempts (observed 4,900+ in 6 minutes)
            if "insufficient_balance" in str(e):
                self.state.balance_backoff_until = time.monotonic() + self.config.balance_backoff_s
                logger.warning(f"Insufficient balance — pausing quoting {self.config.balance_backoff_s:.0f}s")
            else:
                logger.error(f"Order failed: {e}")
            return None

    async def _check_fills_loop(self):
        """Reconcile fills from the exchange every 5 seconds.

        Uses GET /portfolio/fills (one call per cycle) instead of polling each
        order's status. Fills are the authoritative record: they carry the
        actual execution price, size, and the is_taker flag, so the fee we
        book matches what Kalshi actually charged. The old 30s "stale order"
        cancel is deliberately gone — a resting quote at the right price IS
        the strategy; canceling it forfeits queue priority.
        """
        # Look back 60s on the first cycle; the API expects epoch SECONDS.
        last_ts = int(time.time()) - 60

        while self.running:
            await asyncio.sleep(5)
            try:
                # 60s overlap so a slow cycle can't skip fills; dedupe by id.
                response = self.api_client.get_fills(min_ts=last_ts - 60, limit=200)
                for fill in response.get("fills", []):
                    self._process_fill(fill)
                last_ts = int(time.time())
            except Exception as e:
                logger.error(f"Error in fill reconciliation loop: {e}")

    def _process_fill(self, fill: Dict):
        """Book one exchange-reported fill: position, entry price, fee, PnL.

        Kalshi fills carry an action (buy/sell) and a contract side (yes/no).
        We fold both into YES-equivalent terms so inventory is one signed
        number per ticker:
            buy yes / sell no -> long YES  (+1), price = yes_price
            sell yes / buy no -> short YES (-1), price = 1 - no_price
        """
        # Dedupe — fills endpoint returns overlapping windows by design
        fill_id = fill.get("trade_id") or fill.get("fill_id") or \
            f"{fill.get('order_id', '')}-{fill.get('ts', '')}"
        if fill_id in self.state.processed_fill_ids:
            return
        self.state.processed_fill_ids.add(fill_id)

        ticker = fill.get("market_ticker") or fill.get("ticker", "")
        action = fill.get("action", "")
        side = fill.get("side", "")
        if not ticker or action not in ("buy", "sell") or side not in ("yes", "no"):
            return

        # YES-equivalent price and direction
        if side == "yes":
            price_yes = float(fill.get("yes_price_dollars", fill.get("yes_price", 0)) or 0)
        else:
            price_yes = 1.0 - float(fill.get("no_price_dollars", fill.get("no_price", 0)) or 0)
        long_yes = (action == "buy") == (side == "yes")  # True = we got longer YES
        # Kalshi reports fractional fills (count_fp can be 0.5). Ignore sub-1
        # fills entirely — int() flooring them to 0 still bumped counters and
        # fees while changing no position. Position sync reconciles any drift.
        size = int(round(float(fill.get("count_fp", fill.get("count", 1)) or 1)))
        if size <= 0:
            return
        is_taker = bool(fill.get("is_taker", False))
        category = self.active_tickers.get(ticker, ticker.split("-")[0])

        logger.info(f"FILL: {category}/{ticker} {'buy' if long_yes else 'sell'} {size} "
                    f"@ {price_yes * 100:.1f}¢ ({'TAKER' if is_taker else 'maker'})")

        # Fee on EVERY fill at the actual rate the exchange charged. The old
        # code charged maker fee only on closing fills — undercounting both
        # legs and ignoring the 4x taker rate entirely.
        if is_taker:
            fee = compute_taker_fee(size, price_yes, self.config.taker_fee_rate)
            self.state.taker_fills += 1
        else:
            fee = compute_maker_fee(size, price_yes, self.config.maker_fee_rate)
            self.state.maker_fills += 1
        self.state.daily_pnl -= fee
        self.state.cumulative_pnl -= fee
        self.state.fees_paid += fee

        # Update position and weighted avg entry price. Capture the pre-fill
        # entry FIRST — the update below pops it when the fill flattens the
        # position, which would zero out the realized-PnL calculation below.
        # Entry tracking is symmetric for longs AND shorts (the old code never
        # recorded an entry when opening a short, so shorts were invisible to
        # the stop-loss loop and covered with wrong PnL):
        #   flat/flip  -> entry = fill price
        #   extended   -> size-weighted average of old entry and fill price
        #   reduced    -> entry unchanged;  flattened -> entry cleared
        old_inv = self.state.positions[ticker]
        entry_before = self.state.entry_prices.get(ticker, price_yes)
        if long_yes:
            self.state.positions[ticker] += size
            self.state.fills_buy[ticker] += size
        else:
            self.state.positions[ticker] -= size
            self.state.fills_sell[ticker] += size
        new_inv = self.state.positions[ticker]

        if new_inv == 0:
            self.state.entry_prices.pop(ticker, None)
        elif old_inv == 0 or (old_inv > 0) != (new_inv > 0):
            self.state.entry_prices[ticker] = price_yes  # opened or flipped
        elif abs(new_inv) > abs(old_inv):
            # Extended the same direction — weighted average entry
            self.state.entry_prices[ticker] = (
                abs(old_inv) * entry_before + size * price_yes) / abs(new_inv)
        # else: reduced toward zero — entry price unchanged

        self.state.fills += 1
        self.state.record_trade(ticker, "buy" if long_yes else "sell", price_yes * 100, size)

        # A 1-lot order is fully consumed by its fill — release its tracking
        # so the next quote cycle replaces that side instead of skipping
        order_id = fill.get("order_id")
        if order_id:
            self.state.pending_orders.pop(order_id, None)
            active = self.state.active_orders.get(ticker, {})
            for side_key in ("bid_id", "ask_id"):
                if active.get(side_key) == order_id:
                    active[side_key] = None

        # Realized PnL (gross of fees — fees already booked above) when the
        # fill moves inventory toward zero. Uses the PRE-fill entry price.
        realized = 0.0
        if not long_yes and old_inv > 0:      # selling closes (part of) a long
            realized = min(size, old_inv) * (price_yes - entry_before)
        elif long_yes and old_inv < 0:        # buying closes (part of) a short
            realized = min(size, abs(old_inv)) * (entry_before - price_yes)

        if realized != 0.0:
            self.state.daily_pnl += realized
            self.state.cumulative_pnl += realized
            self.state.realized_by_ticker[ticker] += realized  # feeds obs [14]
            logger.info(f"PNL: {ticker} realized ${realized:+.4f} (entry={entry_before:.3f} exit={price_yes:.3f})")

            if realized > 0:
                self.state.wins += 1
                self.state.consecutive_losses[category] = 0
            else:
                self.state.losses += 1
                self.state.consecutive_losses[category] += 1

    async def _sync_positions(self):
        """Sync positions from Kalshi account.

        Reconciles local inventory with the exchange's authoritative state.
        Logs any drift that was detected and corrected.
        """
        try:
            response = self.api_client.get_positions()
            # V2 nests positions under "market_positions" (position_fp strings)
            positions = response.get("market_positions", response.get("positions", []))

            # Build exchange-side position map. The positions payload has used
            # both "ticker" and "market_ticker" across API versions — accept either.
            exchange_positions: Dict[str, int] = {}
            for pos in positions:
                ticker = pos.get("ticker") or pos.get("market_ticker")
                if not ticker:
                    continue
                # V2 reports position_fp (fixed-point string); older shapes used "position"
                position = int(round(float(pos.get("position_fp", pos.get("position", 0)) or 0)))
                exchange_positions[ticker] = position

            # Detect drift: compare local vs exchange for all known tickers
            all_tickers = set(exchange_positions.keys()) | set(
                t for t, v in self.state.positions.items() if v != 0
            )
            drift_count = 0
            for ticker in all_tickers:
                local = self.state.positions.get(ticker, 0)
                remote = exchange_positions.get(ticker, 0)
                if local != remote:
                    logger.warning(
                        f"POSITION DRIFT: {ticker} local={local} exchange={remote}, correcting"
                    )
                    self.state.positions[ticker] = remote
                    drift_count += 1

            # Also set any exchange positions we didn't know about
            for ticker, pos in exchange_positions.items():
                self.state.positions[ticker] = pos

            logger.info(
                f"Synced {len(positions)} positions from Kalshi"
                + (f" ({drift_count} drifts corrected)" if drift_count else "")
            )

        except KalshiAPIError as e:
            logger.error(f"Error syncing positions: {e}")

    async def _position_sync_loop(self):
        """Periodically reconcile local inventory with exchange (every 30s)."""
        while self.running:
            await asyncio.sleep(30)
            await self._sync_positions()

    async def _position_exit_loop(self):
        """Actively unwind positions that are stale, near expiry, or losing.

        Runs every 10 seconds. For each open position, checks:
        1. Stop-loss: unrealized loss exceeds threshold -> market-exit
        2. Expiry: < 2 minutes until market close -> market-exit
        Both send IOC (immediate-or-cancel) orders at a crossing price to
        guarantee fill rather than resting a passive order.
        """
        # Exit when market closes in fewer than this many seconds
        EXPIRY_BUFFER_S = 120
        # Stop-loss floor per contract; the effective threshold widens with the
        # spread (see below) so ordinary bid-ask bounce can't trigger exits
        STOP_LOSS_FLOOR = 0.05

        while self.running:
            await asyncio.sleep(10)

            # Iterate over all tickers with non-zero inventory
            for ticker in list(self.state.positions.keys()):
                inventory = self.state.positions[ticker]
                if inventory == 0:
                    continue

                entry = self.state.entry_prices.get(ticker)
                if entry is None:
                    continue

                # Current mid from the canonical book (skip if unusable)
                book = self.state.orderbooks.get(ticker)
                if book is None or not book.is_valid():
                    continue
                best_bid = book.best_bid()
                best_ask = book.best_ask()
                mid = (best_bid + best_ask) / 2.0
                spread = best_ask - best_bid

                # --- Check stop-loss ---
                # Threshold = max(floor, 2x spread): in a 4c-wide market the mid
                # wobbles 2c just from one side refreshing — exiting on that
                # noise converts temporary marks into realized taker losses
                stop_threshold = max(STOP_LOSS_FLOOR, 2.0 * spread)
                unrealized_per = abs(mid - entry) if inventory != 0 else 0.0
                # Long position losing money: mid dropped below entry
                losing = (inventory > 0 and mid < entry) or (inventory < 0 and mid > entry)
                stop_triggered = losing and unrealized_per >= stop_threshold

                # --- Check expiry ---
                expiry_triggered = False
                close_time = self.state.close_times.get(ticker)
                if close_time:
                    now = datetime.now(close_time.tzinfo)
                    seconds_left = (close_time - now).total_seconds()
                    expiry_triggered = seconds_left < EXPIRY_BUFFER_S

                if not stop_triggered and not expiry_triggered:
                    continue

                reason = "STOP-LOSS" if stop_triggered else "EXPIRY"
                logger.info(
                    f"EXIT ({reason}): {ticker} inv={inventory} entry={entry:.3f} "
                    f"mid={mid:.3f} unrealized={inventory * (mid - entry):+.4f}"
                )

                # Send a crossing IOC order to exit immediately. IOC (not GTC)
                # so an unfilled remainder cancels instead of resting at a
                # price that no longer makes sense.
                # Long -> sell at best_bid. Short -> buy at best_ask.
                try:
                    category = self.active_tickers.get(ticker, "UNKNOWN")
                    if inventory > 0:
                        exit_side, exit_price_cents = "sell", round(best_bid * 100, 1)
                    else:
                        exit_side, exit_price_cents = "buy", round(best_ask * 100, 1)
                    await self._send_order(
                        category=category, ticker=ticker,
                        side=exit_side, price=exit_price_cents,
                        size=abs(inventory),
                        time_in_force="immediate_or_cancel",
                    )
                except Exception as e:
                    logger.error(f"EXIT order failed for {ticker}: {e}")

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
        # Scope the limit to markets this bot quotes — legacy inventory from
        # prior deployments must not block new quoting (it did: $96.50 of old
        # positions vs the $40 cap froze the bot entirely)
        active_value = self.state.position_value(set(self.active_tickers))
        if active_value >= TRADING_CONFIG["max_position_value"]:
            logger.warning(f"Position value limit: ${active_value:.2f}")
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
            asyncio.create_task(self._position_sync_loop()),   # Fix 2: periodic inventory reconciliation
            asyncio.create_task(self._position_exit_loop()),    # Fix 3: stop-loss + expiry exits
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
                # Skip categories without a loaded model
                if category not in self.models:
                    continue
                # Skip halted categories
                if category in self.state.halted_categories:
                    continue

                try:
                    # Fetch orderbook from REST API into the canonical book
                    response = self.api_client.get_orderbook(ticker)
                    book = self.state.orderbooks.setdefault(ticker, LiveBook())
                    book.load_snapshot(response.get("orderbook", {}))

                    if not book.is_valid():
                        continue

                    # Build observation from the canonical book
                    obs = self._build_observation(category, ticker, book)

                    if obs is None:
                        continue

                    # Get model prediction
                    model = self.models[category]
                    action, _ = model.predict(obs, deterministic=True)

                    # Execute action
                    await self._execute_action(category, ticker, action, book)

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

            # Deployment gate: a market-making bot should be ~0% taker. If the
            # taker fraction climbs above 10%, post-only/clamping is not doing
            # its job (or exits dominate) — investigate before scaling up.
            total_tm = self.state.taker_fills + self.state.maker_fills
            taker_frac = self.state.taker_fills / total_tm if total_tm else 0.0
            gate = "PASS" if taker_frac < 0.10 else "FAIL — investigate"

            logger.info("=" * 80)
            logger.info("HOURLY SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Quotes sent: {self.state.quotes_sent}")
            logger.info(f"Fills: {self.state.fills} (Fill rate: {self.state.fill_rate:.1%})")
            logger.info(f"Taker/maker: {self.state.taker_fills}/{self.state.maker_fills} "
                        f"(taker {taker_frac:.1%}) — gate: {gate}")
            logger.info(f"Fees paid: ${self.state.fees_paid:.2f}")
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
        self.subscriptions: Dict[str, callable] = {}  # ticker -> orderbook callback
        self.trade_callback: Optional[callable] = None  # set by LiveTrader for "trade" msgs

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
                # "trade" feeds obs [9]/[16] with real prints (training parity)
                "channels": ["orderbook_delta", "trade"],
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
            logger.debug(f"Processing {msg_type} for {ticker}")
            callback = self.subscriptions[ticker]
            await callback(ticker, data["msg"])
        elif msg_type == "trade" and ticker and self.trade_callback:
            # Public trade print — routed to the trader for obs [9]/[16]
            await self.trade_callback(ticker, data["msg"])
        else:
            logger.debug(f"Ignoring message type '{msg_type}' for ticker '{ticker}' (not subscribed or different type)")


async def main():
    """Main entry point."""
    # One live trader per account: concurrent instances double-quote and
    # cancel each other's orders. The lock handle must outlive main().
    lock_handle = acquire_single_instance_lock()
    if lock_handle is None:
        logger.error("Another live trader instance is already running — exiting. "
                     "(Lock: ~/.kalshi_mm_live.lock)")
        return

    # Check for paper mode flag
    paper_mode = os.getenv("PAPER_MODE", "false").lower() == "true"

    logger.info("=" * 80)
    logger.info("KALSHI LIVE TRADING BOT")
    logger.info("=" * 80)
    logger.info(f"Mode: {'PAPER TRADING' if paper_mode else 'LIVE TRADING'}")
    logger.info(f"Config: {_cli_args.config}")
    logger.info(f"Capital: ${TRADING_CONFIG.get('capital', 0):.2f}")
    logger.info(f"Categories ({len(CATEGORIES)}): {[c.name for c in CATEGORIES]}")
    logger.info(f"Risk Limits: Daily loss ${TRADING_CONFIG.get('max_daily_loss', 0)}, "
                f"Stop loss ${TRADING_CONFIG.get('stop_loss_threshold', 0)}")
    logger.info("=" * 80)

    trader = LiveTrader(paper_mode=paper_mode)
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
