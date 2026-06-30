"""
Kalshi REST API Client
Official API documentation: https://trading-api.readme.io/
"""

import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal, Union
from urllib.parse import urlencode

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class KalshiAPIError(Exception):
    """Kalshi API error."""
    pass


class KalshiRESTClient:
    """Kalshi REST API client for order execution and account management."""

    def __init__(self, api_key: str, api_secret: Union[str, Path], base_url: str = "https://external-api.kalshi.com/trade-api/v2"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.token: Optional[str] = None
        self.token_expires: Optional[datetime] = None

        # Load RSA private key from file or string
        if isinstance(api_secret, (str, Path)):
            secret_path = Path(api_secret)
            if secret_path.exists():
                # Load from file
                with open(secret_path, 'rb') as key_file:
                    self.private_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=None,
                        backend=default_backend()
                    )
            else:
                # Treat as PEM string
                self.private_key = serialization.load_pem_private_key(
                    api_secret.encode() if isinstance(api_secret, str) else api_secret,
                    password=None,
                    backend=default_backend()
                )

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Sign request with RSA-PSS and SHA256 per Kalshi API spec."""
        # Message format: timestamp + method + path (no body for Kalshi)
        message = f"{timestamp}{method}{path}"

        # Sign with RSA-PSS padding (DIGEST_LENGTH salt as per Kalshi docs)
        signature_bytes = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        # Base64 encode the signature
        signature = base64.b64encode(signature_bytes).decode('utf-8')
        return signature

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
        }

        # If we have a valid token, use bearer auth
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            headers["Authorization"] = f"Bearer {self.token}"
        else:
            # Otherwise use API key/signature auth
            timestamp = str(int(time.time() * 1000))
            signature = self._sign_request(timestamp, method, path, body)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        return headers

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Kalshi API."""
        url = f"{self.base_url}{endpoint}"

        # For signing, path must be from API root (/trade-api/v2/...)
        # Extract path from base_url
        from urllib.parse import urlparse
        parsed_base = urlparse(self.base_url)
        base_path = parsed_base.path  # e.g., /trade-api/v2

        # Build query string for GET requests (query params not included in signature)
        if params:
            query_string = "?" + urlencode(params)
            url += query_string

        # Path for signing: full path from API root, no query params
        signing_path = f"{base_path}{endpoint}"

        # Prepare body
        body = ""
        if data:
            body = json.dumps(data)

        headers = self._get_headers(method, signing_path, body)

        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=10)
            elif method == "POST":
                response = self.session.post(url, headers=headers, data=body, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            # Include response body in error message for debugging
            error_detail = ""
            try:
                error_detail = f" - Response: {response.text}"
            except:
                pass
            raise KalshiAPIError(f"API request failed: {e}{error_detail}")
        except requests.exceptions.RequestException as e:
            raise KalshiAPIError(f"API request failed: {e}")

    def login(self, email: str, password: str) -> Dict:
        """Login and get access token."""
        response = self._request("POST", "/login", data={
            "email": email,
            "password": password,
        })

        self.token = response.get("token")
        # Token typically valid for 24 hours
        self.token_expires = datetime.now().replace(hour=23, minute=59, second=59)

        return response

    # === Market Data ===

    def get_markets(self, limit: int = 100, cursor: Optional[str] = None,
                    event_ticker: Optional[str] = None, series_ticker: Optional[str] = None,
                    status: Optional[str] = "open") -> Dict:
        """Get markets."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status

        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> Dict:
        """Get single market by ticker."""
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 5) -> Dict:
        """Get orderbook for a market."""
        return self._request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_trades(self, ticker: Optional[str] = None,
                   min_ts: Optional[int] = None,
                   max_ts: Optional[int] = None,
                   limit: int = 1000,
                   cursor: Optional[str] = None) -> Dict:
        """Get public trades (all participants) for a market."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets/trades", params=params)

    # === Account ===

    def get_balance(self) -> Dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, ticker: Optional[str] = None,
                     settlement_status: Optional[str] = None) -> Dict:
        """Get current positions."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if settlement_status:
            params["settlement_status"] = settlement_status

        return self._request("GET", "/portfolio/positions", params=params)

    def get_fills(self, ticker: Optional[str] = None,
                  order_id: Optional[str] = None,
                  min_ts: Optional[int] = None,
                  max_ts: Optional[int] = None,
                  limit: int = 100) -> Dict:
        """Get fill history."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts

        return self._request("GET", "/portfolio/fills", params=params)

    # === Orders ===

    def create_order(self,
                     ticker: str,
                     side: Literal["bid", "ask"],
                     price_dollars: float,
                     count: float = 1.0,
                     time_in_force: Literal["good_till_canceled", "immediate_or_cancel", "fill_or_kill"] = "good_till_canceled",
                     self_trade_prevention: Literal["taker_at_cross", "maker"] = "taker_at_cross",
                     post_only: bool = False,
                     client_order_id: Optional[str] = None) -> Dict:
        """
        Create a new order (V2 API).

        Args:
            ticker: Market ticker (e.g., "KXAFCCLGAME-24JUN29-B10")
            side: "bid" (buy) or "ask" (sell)
            price_dollars: Limit price in dollars (e.g., 0.32 for 32 cents)
            count: Number of contracts (supports fractional, e.g., 1.0, 10.5)
            time_in_force: Order duration type
            self_trade_prevention: Self-trade prevention type
            post_only: If true, order will only be posted as maker
            client_order_id: Optional custom order ID

        Returns:
            Order confirmation with order_id
        """
        data = {
            "ticker": ticker,
            "side": side,
            # Use 3 decimals for subpenny, 2 decimals for whole-cent prices
            "price": f"{price_dollars:.3f}" if price_dollars != round(price_dollars, 2) else f"{price_dollars:.2f}",
            "count": f"{count:.2f}",  # Fixed-point format with 2 decimals
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention,
            "post_only": post_only,
        }

        if client_order_id:
            data["client_order_id"] = client_order_id
        else:
            data["client_order_id"] = f"mm_{int(time.time() * 1000)}"  # Unique ID

        return self._request("POST", "/portfolio/events/orders", data=data)

    def get_order(self, order_id: str) -> Dict:
        """Get order status (V2 API uses events endpoint)."""
        return self._request("GET", f"/portfolio/events/orders/{order_id}")

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order (V2 API uses events endpoint)."""
        return self._request("DELETE", f"/portfolio/events/orders/{order_id}")

    def get_orders(self,
                   ticker: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 100,
                   cursor: Optional[str] = None) -> Dict:
        """Get orders."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status  # "resting", "canceled", "executed"
        if cursor:
            params["cursor"] = cursor

        return self._request("GET", "/portfolio/orders", params=params)

    def cancel_all_orders(self, ticker: Optional[str] = None) -> Dict:
        """Cancel all open orders (optionally for a specific ticker).

        API V2 doesn't have a bulk cancel endpoint, so we fetch all orders
        and cancel them individually.
        """
        # Get all open orders
        response = self.get_orders(ticker=ticker, status="resting", limit=1000)
        orders = response.get("orders", [])

        canceled_count = 0
        errors = []

        # Cancel each order individually
        for order in orders:
            order_id = order.get("order_id")
            if order_id:
                try:
                    self.cancel_order(order_id)
                    canceled_count += 1
                except Exception as e:
                    errors.append(f"Failed to cancel {order_id}: {e}")

        return {
            "canceled": canceled_count,
            "total": len(orders),
            "errors": errors
        }

    # === Helper Methods ===

    def get_position_for_ticker(self, ticker: str) -> int:
        """Get current position (inventory) for a ticker. Returns net position (positive = long, negative = short)."""
        try:
            response = self.get_positions(ticker=ticker)
            positions = response.get("positions", [])

            if not positions:
                return 0

            # Sum up positions across all contracts for this ticker
            net_position = 0
            for pos in positions:
                if pos["ticker"] == ticker:
                    position = pos.get("position", 0)
                    # Position is typically stored as absolute value with side indicator
                    # Need to check Kalshi's actual response format
                    net_position += position

            return net_position

        except KalshiAPIError:
            return 0

    def place_limit_order(self, ticker: str, side: Literal["buy", "sell"],
                         price_cents: float, size: int = 1) -> Dict:
        """
        Place a limit order (simplified interface).

        Args:
            ticker: Market ticker
            side: "buy" or "sell"
            price_cents: Limit price in cents (1-99), supports subpenny (e.g. 50.1)
            size: Number of contracts

        Returns:
            Order response with order_id
        """
        # Convert cents to dollars and buy/sell to bid/ask
        price_dollars = price_cents / 100.0
        v2_side = "bid" if side == "buy" else "ask"

        return self.create_order(
            ticker=ticker,
            side=v2_side,
            price_dollars=price_dollars,
            count=float(size),
        )

    def get_recent_fills_for_ticker(self, ticker: str, since_minutes: int = 5) -> List[Dict]:
        """Get recent fills for a ticker."""
        min_ts = int((time.time() - since_minutes * 60) * 1000)

        response = self.get_fills(ticker=ticker, min_ts=min_ts)
        return response.get("fills", [])


class KalshiPaperTradingClient(KalshiRESTClient):
    """
    Paper trading client - simulates Kalshi API without real orders.
    Useful for testing before going live.

    NOTE: This inherits from KalshiRESTClient for interface compatibility,
    but overrides all methods that would make real API calls.
    """

    def __init__(self, api_key: str = "paper", api_secret: str = "paper", initial_balance: float = 10000.0):
        # Don't call super().__init__ - paper trading doesn't need real authentication
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"  # For compatibility
        self.session = requests.Session()
        self.paper_balance = initial_balance
        self.paper_positions: Dict[str, int] = {}
        self.paper_orders: Dict[str, Dict] = {}
        self.paper_fills: List[Dict] = []
        self.order_counter = 0
        # Add missing attributes for compatibility with parent class
        self.token = None
        self.token_expires = None
        self.private_key = None

    def _request(self, method: str, path: str, **kwargs) -> Dict:
        """Override _request to prevent real API calls."""
        raise NotImplementedError(
            f"Paper trading client cannot make real API calls. "
            f"Override method for {method} {path} or use simulation methods."
        )

    def get_markets(self, limit: int = 100, cursor: Optional[str] = None,
                   series_ticker: Optional[str] = None,
                   status: Optional[str] = None, **kwargs) -> Dict:
        """
        Fetch REAL market data from live API (read-only operation).
        Paper trading still uses real market data for observations.
        """
        # Create a temporary live client for read-only operations
        # This requires API credentials to be set in environment
        import os
        api_key = os.getenv("KALSHI_API_KEY")
        api_secret = os.getenv("KALSHI_API_SECRET")

        if not api_key or not api_secret:
            logger.info(f"[PAPER] WARNING: No API credentials found. Returning empty markets.")
            logger.info(f"[PAPER] Set KALSHI_API_KEY and KALSHI_API_SECRET to fetch real market data.")
            return {"markets": [], "cursor": None}

        # Create live client just for this read operation
        live_client = KalshiRESTClient(api_key=api_key, api_secret=api_secret)
        return live_client.get_markets(
            limit=limit,
            cursor=cursor,
            series_ticker=series_ticker,
            status=status,
            **kwargs
        )

    def get_orderbook(self, ticker: str, depth: int = 5) -> Dict:
        """
        Fetch REAL orderbook data from live API (read-only operation).
        Paper trading still uses real market data for observations.
        """
        import os
        api_key = os.getenv("KALSHI_API_KEY")
        api_secret = os.getenv("KALSHI_API_SECRET")

        if not api_key or not api_secret:
            logger.info(f"[PAPER] WARNING: No API credentials. Returning simulated orderbook.")
            # Return a wide spread orderbook for testing
            return {
                "orderbook": {
                    "yes": [[1, 100], [2, 100]],
                    "no": [[99, 100], [98, 100]],
                }
            }

        # Create live client just for this read operation
        live_client = KalshiRESTClient(api_key=api_key, api_secret=api_secret)
        return live_client.get_orderbook(ticker=ticker, depth=depth)

    def create_order(self, ticker: str, side: str, price_dollars: float = 0.0,
                     count: float = 1.0, **kwargs) -> Dict:
        """Simulate order creation. Matches KalshiRESTClient.create_order signature."""
        self.order_counter += 1
        order_id = f"paper_order_{self.order_counter}"

        order = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "price": f"{price_dollars:.3f}",
            "count": f"{count:.2f}",
            "status": "resting",
            "created_time": int(time.time() * 1000),
        }

        self.paper_orders[order_id] = order

        logger.info(f"[PAPER] Order: {order_id} {side} {count:.0f} {ticker} @ {price_dollars:.3f}")

        return {"order": order, "order_id": order_id}

    def get_order(self, order_id: str) -> Dict:
        """Get simulated order."""
        order = self.paper_orders.get(order_id)
        if not order:
            raise KalshiAPIError(f"Order not found: {order_id}")
        return {"order": order}

    def cancel_order(self, order_id: str) -> Dict:
        """Simulate order cancellation."""
        if order_id in self.paper_orders:
            self.paper_orders[order_id]["status"] = "canceled"
            logger.info(f"[PAPER] Order canceled: {order_id}")
        return {"order_id": order_id, "status": "canceled"}

    def get_balance(self) -> Dict:
        """Get simulated balance."""
        return {
            "balance": int(self.paper_balance * 100),  # cents
            "payout": 0,
        }

    def get_positions(self, ticker: Optional[str] = None, **kwargs) -> Dict:
        """Get simulated positions."""
        positions = []
        for tick, pos in self.paper_positions.items():
            if ticker is None or tick == ticker:
                positions.append({
                    "ticker": tick,
                    "position": pos,
                })
        return {"positions": positions}

    def get_position_for_ticker(self, ticker: str) -> int:
        """Get simulated position."""
        return self.paper_positions.get(ticker, 0)

    def simulate_fill(self, order_id: str, fill_price: int, fill_count: int):
        """Simulate an order fill (for testing)."""
        order = self.paper_orders.get(order_id)
        if not order:
            return

        ticker = order["ticker"]
        action = order["action"]
        side = order["side"]

        # Update position
        if action == "buy":
            self.paper_positions[ticker] = self.paper_positions.get(ticker, 0) + fill_count
            cost = fill_price * fill_count
            self.paper_balance -= cost / 100.0
        else:  # sell
            self.paper_positions[ticker] = self.paper_positions.get(ticker, 0) - fill_count
            proceeds = fill_price * fill_count
            self.paper_balance += proceeds / 100.0

        # Record fill
        fill = {
            "fill_id": f"fill_{len(self.paper_fills) + 1}",
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": fill_count,
            "price": fill_price,
            "created_time": int(time.time() * 1000),
        }
        self.paper_fills.append(fill)

        # Update order status
        order["status"] = "executed"
        order["remaining_count"] = order.get("count", 0) - fill_count

        logger.info(f"[PAPER] Fill executed: {action} {fill_count} {ticker} @ {fill_price}¢")

    def get_fills(self, **kwargs) -> Dict:
        """Get simulated fills."""
        return {"fills": self.paper_fills}


if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.getenv("KALSHI_API_KEY", "demo_key")
    api_secret = os.getenv("KALSHI_API_SECRET", "demo_secret")

    # Paper trading mode
    print("=== Paper Trading Mode ===")
    client = KalshiPaperTradingClient(api_key, api_secret, initial_balance=1000.0)

    # Simulate some orders
    order1 = client.create_order(
        ticker="KXAFCCLGAME-24JUN29-B10",
        side="yes",
        action="buy",
        count=1,
        type="limit",
        yes_price=50,
    )
    print(f"Order created: {order1['order_id']}")

    # Simulate a fill
    client.simulate_fill(order1["order_id"], fill_price=50, fill_count=1)

    # Check position
    position = client.get_position_for_ticker("KXAFCCLGAME-24JUN29-B10")
    print(f"Position: {position}")

    # Check balance
    balance = client.get_balance()
    print(f"Balance: ${balance['balance'] / 100:.2f}")
