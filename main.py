import asyncio
import logging

from config import (
    API_KEY, KEY_PATH, PAPER_TRADING,
    VOL_RATIO_THRESHOLD, RANGE_THRESHOLD, MIN_EDGE, MIN_VOLUME,
    BUY_PROB_THRESHOLD, SELL_PROB_THRESHOLD, EDGE_VOL_THRESHOLD,
)
from market_filter import is_btc_market, register_market
from volatility import VolatilityTracker
from model.model_interface import ModelInterface
from quoting_engine import generate_quotes, format_quotes
from authentication_to_kalshi.websocket_client import KalshiWebSocket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("kalshi")

vol_trackers = {}
trade_counts = {}
orderbooks = {}
inventory = {}
model = ModelInterface()


def _get_tracker(ticker):
    if ticker not in vol_trackers:
        vol_trackers[ticker] = VolatilityTracker()
        trade_counts[ticker] = 0
        orderbooks[ticker] = {"yes": [], "no": []}
        inventory[ticker] = 0
    return vol_trackers[ticker]


def on_ticker(msg):
    data = msg.get("msg", {})
    ticker = data.get("market_ticker", "")
    if not is_btc_market(ticker):
        return

    if register_market(ticker):
        log.info("Discovered BTC market: %s", ticker)

    price = data.get("yes_ask") or data.get("last_price")
    if price is None:
        return

    price = price / 100 if price > 1 else price
    tracker = _get_tracker(ticker)
    tracker.update(price)
    _maybe_quote(ticker)


def on_trade(msg):
    data = msg.get("msg", {})
    ticker = data.get("market_ticker", "")
    if not is_btc_market(ticker):
        return

    price = data.get("yes_price", 0)
    price = price / 100 if price > 1 else price
    tracker = _get_tracker(ticker)
    tracker.update(price)
    trade_counts[ticker] = trade_counts.get(ticker, 0) + data.get("count", 1)
    _maybe_quote(ticker)


def on_orderbook_delta(msg):
    data = msg.get("msg", {})
    ticker = data.get("market_ticker", "")
    if not is_btc_market(ticker):
        return

    _get_tracker(ticker)
    if "yes" in data:
        orderbooks[ticker]["yes"] = data["yes"]
    if "no" in data:
        orderbooks[ticker]["no"] = data["no"]


def _maybe_quote(ticker):
    tracker = vol_trackers.get(ticker)
    if not tracker:
        return

    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]
    if market_price <= 0:
        return

    if metrics["vol_ratio"] < VOL_RATIO_THRESHOLD:
        return
    if metrics["range"] < RANGE_THRESHOLD:
        return

    recent_volume = trade_counts.get(ticker, 0)
    if recent_volume < MIN_VOLUME:
        return

    if not model.is_ready():
        return

    features = {"market_price": market_price, **metrics}
    pred = model.predict(ticker, features)

    edge = abs(pred - market_price)
    if edge < MIN_EDGE:
        return

    if not (pred > BUY_PROB_THRESHOLD or pred < SELL_PROB_THRESHOLD):
        return

    if edge * metrics["vol_ratio"] < EDGE_VOL_THRESHOLD:
        return

    prices = list(tracker.prices)
    quotes = generate_quotes(
        model_prob=pred,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=recent_volume,
        prices=prices,
        inventory=inventory.get(ticker, 0),
    )

    if quotes["buy"] or quotes["sell"]:
        log.info("\n%s", format_quotes(quotes, ticker))
        if not PAPER_TRADING:
            log.info("LIVE mode — order execution hook placeholder for %s", ticker)


async def main():
    if not API_KEY:
        log.error("PROD_API_KEY not set in .env")
        return
    if not KEY_PATH:
        log.error("PROD_KEY_PATH not set in .env")
        return

    mode = "PAPER" if PAPER_TRADING else "LIVE"
    log.info("Starting Kalshi BTC trading system (%s mode)", mode)

    ws = KalshiWebSocket(
        on_ticker=on_ticker,
        on_trade=on_trade,
        on_orderbook_delta=on_orderbook_delta,
    )
    await ws.run()


if __name__ == "__main__":
    asyncio.run(main())
