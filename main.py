"""
Kalshi BTC Market Making Bot - Main Entry Point

Real-time market making system for Kalshi prediction markets.

ARCHITECTURE:
1. WebSocket connection receives live market data (ticker, trades, orderbook)
2. Market filter identifies BTC markets matching our criteria
3. Volatility tracker maintains rolling window of price metrics
4. Model predicts fair value for each market
5. Multi-stage filter pipeline identifies trading opportunities
6. Quoting engine generates optimal limit orders
7. Orders logged (paper trading) or submitted to Kalshi (live trading)

FILTER PIPELINE (all must pass):
├─ Volatility spike (vol_ratio >= threshold)
├─ Sufficient price range
├─ Adequate trading volume
├─ Model has sufficient edge
├─ Strong directional signal (very bullish or bearish)
└─ Combined edge × volatility threshold

Run with: python main.py
Test concepts in: test_of_concept/
"""
import asyncio
import logging
from typing import Dict

from config import (
    API_KEY, KEY_PATH, PAPER_TRADING,
    VOL_RATIO_THRESHOLD, RANGE_THRESHOLD, MIN_EDGE, MIN_VOLUME,
    BUY_PROB_THRESHOLD, SELL_PROB_THRESHOLD, EDGE_VOL_THRESHOLD,
)
from market_selector.market_filter import is_btc_market, register_market
from volatility import VolatilityTracker
from model.model_interface import ModelInterface
from market_selector.quoting_engine import generate_quotes, format_quotes
from authentication_to_kalshi.websocket_client import KalshiWebSocket

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kalshi")

# ============================================================================
# GLOBAL STATE
# ============================================================================
# Maintains per-market state across WebSocket message handlers

# Volatility trackers for each market (ticker -> VolatilityTracker)
vol_trackers: Dict[str, VolatilityTracker] = {}

# Recent trade counts for each market (ticker -> count)
trade_counts: Dict[str, int] = {}

# Current orderbooks for each market (ticker -> {"yes": [...], "no": [...]})
orderbooks: Dict[str, Dict] = {}

# Current inventory for each market (ticker -> position)
# Positive = long, negative = short, zero = flat
inventory: Dict[str, int] = {}

# Prediction model (shared across all markets)
model = ModelInterface()


def _get_tracker(ticker: str) -> VolatilityTracker:
    """
    Get or create volatility tracker for a market.

    Lazy initialization: creates tracker and supporting state on first access.
    This avoids memory waste for markets we never trade.

    Args:
        ticker: Market ticker symbol

    Returns:
        VolatilityTracker instance for this market
    """
    if ticker not in vol_trackers:
        # First time seeing this market - initialize all state
        vol_trackers[ticker] = VolatilityTracker()
        trade_counts[ticker] = 0
        orderbooks[ticker] = {"yes": [], "no": []}
        inventory[ticker] = 0

    return vol_trackers[ticker]


# ============================================================================
# WEBSOCKET MESSAGE HANDLERS
# ============================================================================
# Callbacks invoked when market data arrives from Kalshi WebSocket


def on_ticker(msg: Dict) -> None:
    """
    Handle ticker message (best bid/ask update for a market).

    Ticker messages arrive frequently (~1-2 per second per active market).
    We use them to:
    1. Discover new BTC markets
    2. Update volatility tracker with latest price
    3. Potentially generate new quotes

    Args:
        msg: WebSocket message with structure:
            {"type": "ticker", "msg": {"market_ticker": "...", "yes_ask": ..., ...}}
    """
    data = msg.get("msg", {})
    ticker = data.get("market_ticker", "")

    # Filter: only process BTC markets
    if not is_btc_market(ticker):
        return

    # Log newly discovered markets
    if register_market(ticker):
        log.info("Discovered BTC market: %s", ticker)

    # Extract price (prefer yes_ask, fallback to last_price)
    price = data.get("yes_ask") or data.get("last_price")
    if price is None:
        return

    # Kalshi sends prices in cents (0-100) or decimal (0.0-1.0)
    # Normalize to decimal
    price = price / 100 if price > 1 else price

    # Update volatility tracker with new price
    tracker = _get_tracker(ticker)
    tracker.update(price)

    # Check if we should generate quotes
    _maybe_quote(ticker)


def on_trade(msg: Dict) -> None:
    """
    Handle trade execution message.

    Trade messages indicate actual fills occurred in the market.
    Higher trade volume suggests more liquidity and fill probability.

    Args:
        msg: WebSocket message with structure:
            {"type": "trade", "msg": {"market_ticker": "...", "yes_price": ..., "count": ...}}
    """
    data = msg.get("msg", {})
    ticker = data.get("market_ticker", "")

    # Filter: only process BTC markets
    if not is_btc_market(ticker):
        return

    # Extract trade price and normalize to decimal
    price = data.get("yes_price", 0)
    price = price / 100 if price > 1 else price

    # Update volatility tracker with trade price
    tracker = _get_tracker(ticker)
    tracker.update(price)

    # Increment trade counter for volume filtering
    trade_counts[ticker] = trade_counts.get(ticker, 0) + data.get("count", 1)

    # Check if we should generate quotes
    _maybe_quote(ticker)


def on_orderbook_delta(msg: Dict) -> None:
    """
    Handle orderbook change message (bids/asks updated).

    Orderbook deltas show current limit order depth on each side.
    Currently just stored for potential future use (e.g., spread analysis).

    Args:
        msg: WebSocket message with structure:
            {"type": "orderbook_delta", "msg": {"market_ticker": "...", "yes": [...], "no": [...]}}
    """
    data = msg.get("msg", {})
    ticker = data.get("market_ticker", "")

    # Filter: only process BTC markets
    if not is_btc_market(ticker):
        return

    # Ensure market state exists
    _get_tracker(ticker)

    # Update stored orderbook (for future use)
    if "yes" in data:
        orderbooks[ticker]["yes"] = data["yes"]
    if "no" in data:
        orderbooks[ticker]["no"] = data["no"]


# ============================================================================
# QUOTE GENERATION LOGIC
# ============================================================================


def _maybe_quote(ticker: str) -> None:
    """
    Evaluate whether to generate quotes for a market.

    Implements multi-stage filter pipeline to identify high-quality
    trading opportunities. Each filter eliminates markets that don't
    meet our criteria, saving computation on quote generation.

    FILTER PIPELINE:
    1. Valid price data exists
    2. Volatility spike detected (vol_ratio >= threshold)
    3. Sufficient price range (not stagnant)
    4. Adequate trading volume (liquid market)
    5. Model is ready to predict
    6. Sufficient edge (model disagrees with market)
    7. Strong directional signal (very bullish or bearish)
    8. Combined edge × volatility threshold

    If all filters pass, generates optimal quotes and logs them
    (or submits to Kalshi if PAPER_TRADING=false).

    Args:
        ticker: Market ticker to evaluate
    """
    # Get volatility tracker (returns None if market never seen)
    tracker = vol_trackers.get(ticker)
    if not tracker:
        return

    # Calculate current market metrics
    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]

    # Filter 1: Valid price data
    if market_price <= 0:
        return

    # Filter 2: Volatility spike detected
    # Recent volatility must be significantly higher than average
    # Indicates market is active and mispricing is more likely
    if metrics["vol_ratio"] < VOL_RATIO_THRESHOLD:
        return

    # Filter 3: Sufficient price range
    # Market must have moved enough to indicate real activity
    # Eliminates stagnant markets with no trading opportunity
    if metrics["range"] < RANGE_THRESHOLD:
        return

    # Filter 4: Adequate trading volume
    # Must have seen enough recent trades to ensure liquidity
    # Low volume = higher risk of adverse selection
    recent_volume = trade_counts.get(ticker, 0)
    if recent_volume < MIN_VOLUME:
        return

    # Filter 5: Model ready
    # Don't trade if model hasn't finished initializing
    if not model.is_ready():
        return

    # Get model prediction of fair value
    features = {"market_price": market_price, **metrics}
    pred = model.predict(ticker, features)

    # Filter 6: Sufficient edge
    # Model must disagree with market by at least MIN_EDGE
    # Ensures profit potential justifies transaction costs
    edge = abs(pred - market_price)
    if edge < MIN_EDGE:
        return

    # Filter 7: Strong directional signal
    # Only trade when model has high conviction
    # Weak signals (near 0.5) filtered out
    if not (pred > BUY_PROB_THRESHOLD or pred < SELL_PROB_THRESHOLD):
        return

    # Filter 8: Combined edge × volatility threshold
    # High edge + high volatility = best opportunities
    # Eliminates marginal opportunities
    if edge * metrics["vol_ratio"] < EDGE_VOL_THRESHOLD:
        return

    # All filters passed - generate optimal quotes
    prices = list(tracker.prices)
    quotes = generate_quotes(
        model_prob=pred,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=recent_volume,
        prices=prices,
        inventory=inventory.get(ticker, 0),
    )

    # Log and potentially submit quotes
    if quotes["buy"] or quotes["sell"]:
        # Always log the quote for visibility
        log.info("\n%s", format_quotes(quotes, ticker))

        # Execute order if in live trading mode
        if not PAPER_TRADING:
            # TODO: Implement actual order submission via Kalshi REST API
            # POST /trade-api/v2/portfolio/orders
            # Payload: {ticker, side, type: "limit", yes_price/no_price, count}
            log.info("LIVE mode — order execution hook placeholder for %s", ticker)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main() -> None:
    """
    Main entry point for the trading system.

    Validates configuration, initializes WebSocket connection,
    and runs the event loop until terminated.

    The system will:
    1. Connect to Kalshi WebSocket API
    2. Subscribe to ticker, trade, and orderbook channels
    3. Process incoming messages through event handlers
    4. Generate quotes when filter criteria are met
    5. Log quotes (paper trading) or submit orders (live trading)
    6. Automatically reconnect if connection drops

    Runs indefinitely until process is killed (Ctrl+C).
    """
    # Validate required configuration
    if not API_KEY:
        log.error("PROD_API_KEY not set in .env")
        return
    if not KEY_PATH:
        log.error("PROD_KEY_PATH not set in .env")
        return

    # Display trading mode for safety
    mode = "PAPER" if PAPER_TRADING else "LIVE"
    log.info("Starting Kalshi BTC trading system (%s mode)", mode)

    # Create WebSocket client with our event handlers
    ws = KalshiWebSocket(
        on_ticker=on_ticker,
        on_trade=on_trade,
        on_orderbook_delta=on_orderbook_delta,
    )

    # Run WebSocket connection loop (never returns)
    # Handles automatic reconnection on disconnect
    await ws.run()


if __name__ == "__main__":
    # Entry point: start async event loop
    asyncio.run(main())
