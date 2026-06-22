import asyncio
import logging
from collections import deque

import httpx

log = logging.getLogger(__name__)

# Exchange API endpoints (same as model/tft/data_fetcher.py)
_COINBASE_BASE = "https://api.exchange.coinbase.com"
_KRAKEN_BASE = "https://api.kraken.com"
_BYBIT_BASE = "https://api.bybit.com"
_BINANCE_BASE = "https://api.binance.com"
_BYBIT_FUTURES = "https://api.bybit.com"
_BINANCE_FUTURES = "https://fapi.binance.com"
_TIMEOUT = 10.0

# Rolling window sizes for returns
_SAMPLES_5M = 10    # 10 samples * 30s = 5 minutes
_SAMPLES_1H = 120   # 120 samples * 30s = 60 minutes
_MAX_HISTORY = 130   # keep a few extra beyond 1h window


class BTCDataPoller:
    """Polls BTC spot price and funding rate from external exchanges.

    Maintains a rolling deque of prices for computing returns.
    Uses the same fallback chain as model/tft/data_fetcher.py:
      Spot: Coinbase -> Kraken -> Bybit -> Binance
      Funding: Bybit -> Binance
    """

    def __init__(self, poll_interval_s: float = 30.0) -> None:
        self._interval = poll_interval_s
        # Deque with max length automatically discards oldest on append
        self._prices: deque[float] = deque(maxlen=_MAX_HISTORY)
        self._session_start: float = 0.0
        self._latest_funding: float = 0.0
        self._running = False

    def spot_price(self) -> float:
        """Latest BTC/USD spot price, 0.0 if unavailable."""
        return self._prices[-1] if self._prices else 0.0

    def session_start_price(self) -> float:
        """BTC price at session start, for log-normalization."""
        return self._session_start

    def return_5m(self) -> float:
        """5-minute BTC return, 0.0 if insufficient data.

        Computes (current - price_from_5m_ago) / price_from_5m_ago.
        Returns 0.0 if we don't have at least 11 samples (10 intervals).
        """
        # Need > _SAMPLES_5M samples to compute return over the 5m window
        if len(self._prices) <= _SAMPLES_5M:
            return 0.0
        # Price from 10 samples ago (5 minutes back)
        old = self._prices[-1 - _SAMPLES_5M]
        if old == 0.0:
            return 0.0
        # Log return: (current - old) / old
        return (self._prices[-1] - old) / old

    def return_1h(self) -> float:
        """1-hour BTC return, 0.0 if insufficient data.

        Computes (current - price_from_1h_ago) / price_from_1h_ago.
        Returns 0.0 if we don't have at least 121 samples (120 intervals).
        """
        # Need > _SAMPLES_1H samples to compute return over the 1h window
        if len(self._prices) <= _SAMPLES_1H:
            return 0.0
        # Price from 120 samples ago (60 minutes back)
        old = self._prices[-1 - _SAMPLES_1H]
        if old == 0.0:
            return 0.0
        # Log return: (current - old) / old
        return (self._prices[-1] - old) / old

    def funding_rate(self) -> float:
        """Latest perpetual funding rate, 0.0 if unavailable."""
        return self._latest_funding

    # -- Internal update methods (also used by tests) --

    def _on_spot_update(self, price: float) -> None:
        """Record a new spot price observation.

        Sets session_start_price on first update.
        """
        if self._session_start == 0.0:
            self._session_start = price
        self._prices.append(price)

    def _on_funding_update(self, rate: float) -> None:
        """Record a new funding rate observation."""
        self._latest_funding = rate

    # -- Async polling loop --

    async def start(self) -> None:
        """Start the background polling loop. Runs until stop() is called."""
        self._running = True
        while self._running:
            await self._poll_once()
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        """Signal the polling loop to stop."""
        self._running = False

    async def _poll_once(self) -> None:
        """Fetch spot price and funding rate with fallback chains."""
        # Spot price fallback chain
        spot = await self._fetch_spot()
        if spot is not None:
            self._on_spot_update(spot)

        # Funding rate fallback chain
        funding = await self._fetch_funding()
        if funding is not None:
            self._on_funding_update(funding)

    async def _fetch_spot(self) -> float | None:
        """Try exchanges in order until one returns a valid price.

        Fallback chain: Coinbase -> Kraken -> Bybit -> Binance
        Returns None if all providers fail.
        """
        fetchers = [
            ("coinbase", self._spot_coinbase),
            ("kraken", self._spot_kraken),
            ("bybit", self._spot_bybit),
            ("binance", self._spot_binance),
        ]
        for name, fn in fetchers:
            try:
                price = await fn()
                if price is not None and price > 0:
                    return price
            except Exception as exc:
                log.debug("Spot %s failed: %s", name, exc)
        log.warning("All spot providers failed")
        return None

    async def _fetch_funding(self) -> float | None:
        """Try funding rate exchanges in order.

        Fallback chain: Bybit -> Binance
        Returns None if all providers fail.
        """
        fetchers = [
            ("bybit", self._funding_bybit),
            ("binance", self._funding_binance),
        ]
        for name, fn in fetchers:
            try:
                rate = await fn()
                if rate is not None:
                    return rate
            except Exception as exc:
                log.debug("Funding %s failed: %s", name, exc)
        return None

    # -- Individual exchange fetchers --

    async def _spot_coinbase(self) -> float | None:
        """Get BTC spot from Coinbase ticker endpoint."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_COINBASE_BASE}/products/BTC-USD/ticker")
            resp.raise_for_status()
            return float(resp.json()["price"])

    async def _spot_kraken(self) -> float | None:
        """Get BTC spot from Kraken ticker endpoint.

        Kraken returns {"result": {"XXBTZUSD": {"c": ["price", "volume"]}}}.
        """
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_KRAKEN_BASE}/0/public/Ticker", params={"pair": "XBTUSD"}
            )
            resp.raise_for_status()
            data = resp.json()
            # Iterate through result pairs looking for price
            for pair_data in data.get("result", {}).values():
                if "c" in pair_data:
                    return float(pair_data["c"][0])
        return None

    async def _spot_bybit(self) -> float | None:
        """Get BTC spot from Bybit ticker endpoint."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BYBIT_BASE}/v5/market/tickers",
                params={"category": "spot", "symbol": "BTCUSDT"},
            )
            resp.raise_for_status()
            items = resp.json().get("result", {}).get("list", [])
            if items:
                return float(items[0]["lastPrice"])
        return None

    async def _spot_binance(self) -> float | None:
        """Get BTC spot from Binance ticker endpoint."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BINANCE_BASE}/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
            )
            resp.raise_for_status()
            return float(resp.json()["price"])

    async def _funding_bybit(self) -> float | None:
        """Get latest funding rate from Bybit linear (perpetual) market."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BYBIT_FUTURES}/v5/market/tickers",
                params={"category": "linear", "symbol": "BTCUSDT"},
            )
            resp.raise_for_status()
            items = resp.json().get("result", {}).get("list", [])
            if items:
                return float(items[0].get("fundingRate", 0))
        return None

    async def _funding_binance(self) -> float | None:
        """Get latest funding rate from Binance futures market."""
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{_BINANCE_FUTURES}/fapi/v1/premiumIndex",
                params={"symbol": "BTCUSDT"},
            )
            resp.raise_for_status()
            return float(resp.json().get("lastFundingRate", 0))
