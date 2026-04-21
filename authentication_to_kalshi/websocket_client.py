"""
WebSocket client for real-time Kalshi market data.

Maintains persistent connection to Kalshi's WebSocket API to receive:
- Ticker updates (best bid/ask prices across all markets)
- Trade executions
- Orderbook changes

Implements exponential backoff reconnection strategy for reliability.
"""
import asyncio
import json
import logging
from typing import Callable, Optional, Set

import websockets
from websockets.client import WebSocketClientProtocol

from authentication_to_kalshi.auth import load_private_key, make_ws_headers
from config import API_KEY, KEY_PATH, WS_URL, RECONNECT_BASE, RECONNECT_MAX

log = logging.getLogger(__name__)

# Message IDs for WebSocket commands
MSG_ID_GLOBAL_TICKER = 1
MSG_ID_MARKET_SUBSCRIBE = 2


class KalshiWebSocket:
    """
    Manages WebSocket connection to Kalshi with automatic reconnection.

    Subscribes to:
    1. Global ticker channel (all market price updates)
    2. Per-market channels (trades, orderbook) for discovered markets

    Callbacks are invoked when messages arrive, allowing caller to
    process data without blocking the WebSocket receive loop.
    """

    def __init__(
        self,
        on_ticker: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
        on_orderbook_delta: Optional[Callable] = None
    ) -> None:
        """
        Initialize WebSocket client with message handlers.

        Args:
            on_ticker: Callback for ticker messages (price updates)
            on_trade: Callback for trade execution messages
            on_orderbook_delta: Callback for orderbook change messages
        """
        # Callback functions for different message types
        self.on_ticker = on_ticker
        self.on_trade = on_trade
        self.on_orderbook_delta = on_orderbook_delta

        # WebSocket connection (None when disconnected)
        self._ws: Optional[WebSocketClientProtocol] = None

        # Load private key for authentication
        self._key = load_private_key(KEY_PATH)

        # Track markets we've subscribed to (for re-subscription after reconnect)
        self._subscribed: Set[str] = set()

    async def connect(self) -> None:
        """
        Establish authenticated WebSocket connection to Kalshi.

        Creates signed headers using RSA private key, connects to WebSocket,
        and re-subscribes to all previously subscribed channels.
        """
        # Generate authentication headers with signature
        headers = make_ws_headers(API_KEY, self._key)

        # Connect to Kalshi WebSocket API
        self._ws = await websockets.connect(WS_URL, additional_headers=headers)
        log.info("Connected to %s", WS_URL)

        # Subscribe to global ticker feed (all markets)
        await self._subscribe_global()

        # Re-subscribe to specific markets (after reconnection)
        for ticker in list(self._subscribed):
            await self._send_subscribe(ticker)

    async def _subscribe_global(self) -> None:
        """
        Subscribe to global ticker channel for all market price updates.

        This channel provides best bid/ask for all active markets.
        Used to discover new markets matching our filter criteria.
        """
        msg = {
            "id": MSG_ID_GLOBAL_TICKER,
            "cmd": "subscribe",
            "params": {"channels": ["ticker"]}
        }
        await self._ws.send(json.dumps(msg))
        log.info("Subscribed to global ticker channel")

    async def subscribe_market(self, ticker: str) -> None:
        """
        Subscribe to detailed market data for a specific ticker.

        Subscribes to:
        - Trade executions
        - Orderbook deltas (bid/ask changes)

        Args:
            ticker: Market ticker to subscribe to
        """
        # Avoid duplicate subscriptions
        if ticker in self._subscribed:
            return

        # Track subscription for reconnection
        self._subscribed.add(ticker)

        # Send subscription if connected
        if self._ws:
            await self._send_subscribe(ticker)

    async def _send_subscribe(self, ticker: str) -> None:
        """
        Send market-specific subscription command.

        Args:
            ticker: Market ticker to subscribe to
        """
        msg = {
            "id": MSG_ID_MARKET_SUBSCRIBE,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta", "trade"],
                "market_tickers": [ticker],
            },
        }
        await self._ws.send(json.dumps(msg))
        log.info("Subscribed to market %s", ticker)

    async def listen(self) -> None:
        """
        Listen for incoming WebSocket messages and dispatch to handlers.

        Runs until connection closes. Parses JSON messages and routes
        to appropriate callback based on message type.
        """
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                # Ignore malformed messages
                continue

            # Route message to appropriate handler
            msg_type = msg.get("type")
            if msg_type == "ticker" and self.on_ticker:
                self.on_ticker(msg)
            elif msg_type == "trade" and self.on_trade:
                self.on_trade(msg)
            elif msg_type == "orderbook_delta" and self.on_orderbook_delta:
                self.on_orderbook_delta(msg)

    async def run(self) -> None:
        """
        Main connection loop with exponential backoff reconnection.

        Continuously maintains connection to Kalshi. If connection drops,
        automatically reconnects with exponential backoff delay.

        Never returns - runs until process is terminated.
        """
        delay = RECONNECT_BASE

        while True:
            try:
                # Connect and start listening
                await self.connect()

                # Reset delay on successful connection
                delay = RECONNECT_BASE

                # Listen for messages (blocks until disconnect)
                await self.listen()

            except (websockets.ConnectionClosed, OSError) as e:
                # Expected disconnection - reconnect with backoff
                log.warning("Connection lost: %s. Reconnecting in %ds...", e, delay)
                await asyncio.sleep(delay)

                # Exponential backoff, capped at maximum
                delay = min(delay * 2, RECONNECT_MAX)

            except Exception:
                # Unexpected error - log and retry with backoff
                log.exception("Unexpected error in WebSocket loop")
                await asyncio.sleep(delay)

                # Exponential backoff, capped at maximum
                delay = min(delay * 2, RECONNECT_MAX)
