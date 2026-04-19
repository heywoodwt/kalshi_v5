import asyncio
import json
import logging

import websockets

from authentication_to_kalshi.auth import load_private_key, make_ws_headers
from config import API_KEY, KEY_PATH, WS_URL, RECONNECT_BASE, RECONNECT_MAX

log = logging.getLogger(__name__)


class KalshiWebSocket:
    def __init__(self, on_ticker=None, on_trade=None, on_orderbook_delta=None):
        self.on_ticker = on_ticker
        self.on_trade = on_trade
        self.on_orderbook_delta = on_orderbook_delta
        self._ws = None
        self._key = load_private_key(KEY_PATH)
        self._subscribed = set()

    async def connect(self):
        headers = make_ws_headers(API_KEY, self._key)
        self._ws = await websockets.connect(WS_URL, additional_headers=headers)
        log.info("Connected to %s", WS_URL)
        await self._subscribe_global()
        for ticker in list(self._subscribed):
            await self._send_subscribe(ticker)

    async def _subscribe_global(self):
        msg = {"id": 1, "cmd": "subscribe", "params": {"channels": ["ticker"]}}
        await self._ws.send(json.dumps(msg))
        log.info("Subscribed to global ticker channel")

    async def subscribe_market(self, ticker):
        if ticker in self._subscribed:
            return
        self._subscribed.add(ticker)
        if self._ws:
            await self._send_subscribe(ticker)

    async def _send_subscribe(self, ticker):
        msg = {
            "id": 2,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "trade"],
                "market_tickers": [ticker],
            },
        }
        await self._ws.send(json.dumps(msg))
        log.info("Subscribed to market %s", ticker)

    async def listen(self):
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type == "ticker" and self.on_ticker:
                self.on_ticker(msg)
            elif msg_type == "trade" and self.on_trade:
                self.on_trade(msg)
            elif msg_type == "orderbook_delta" and self.on_orderbook_delta:
                self.on_orderbook_delta(msg)

    async def run(self):
        delay = RECONNECT_BASE
        while True:
            try:
                await self.connect()
                delay = RECONNECT_BASE
                await self.listen()
            except (websockets.ConnectionClosed, OSError) as e:
                log.warning("Connection lost: %s. Reconnecting in %ds...", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)
            except Exception:
                log.exception("Unexpected error in WebSocket loop")
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)
