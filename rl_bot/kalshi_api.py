"""
Kalshi REST API Client
Official API documentation: https://trading-api.readme.io/
"""

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Literal
from urllib.parse import urlencode

import requests


class KalshiAPIError(Exception):
    """Kalshi API error."""
    pass


class KalshiRESTClient:
    """Kalshi REST API client for order execution and account management."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://trading-api.kalshi.com/trade-api/v2"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.token: Optional[str] = None
        self.token_expires: Optional[datetime] = None

    def _sign_request(self, method: str, path: str, body: str = "") -> str:
        """Sign request with HMAC-SHA256."""
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
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
            signature = self._sign_request(method, path, body)
            headers["KALSHI-ACCESS-KEY"] = self.api_key
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp

        return headers

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to Kalshi API."""
        url = f"{self.base_url}{endpoint}"

        # Build query string for GET requests
        if params:
            query_string = "?" + urlencode(params)
            url += query_string
            path = endpoint + query_string
        else:
            path = endpoint

        # Prepare body
        body = ""
        if data:
            body = json.dumps(data)

        headers = self._get_headers(method, path, body)

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
                     side: Literal["yes", "no"],
                     action: Literal["buy", "sell"],
                     count: int,
                     type: Literal["market", "limit"] = "limit",
                     yes_price: Optional[int] = None,
                     no_price: Optional[int] = None,
                     expiration_ts: Optional[int] = None,
                     sell_position_floor: Optional[int] = None,
                     buy_max_cost: Optional[int] = None) -> Dict:
        """
        Create a new order.

        Args:
            ticker: Market ticker (e.g., "KXAFCCLGAME-24JUN29-B10")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type: "market" or "limit"
            yes_price: Price in cents (1-99) for yes side limit orders
            no_price: Price in cents (1-99) for no side limit orders
            expiration_ts: Unix timestamp in milliseconds for order expiration
            sell_position_floor: Minimum position to maintain when selling
            buy_max_cost: Maximum cost in cents for buy orders

        Returns:
            Order confirmation with order_id
        """
        data = {
            "ticker": ticker,
            "client_order_id": f"mm_{int(time.time() * 1000)}",  # Unique ID
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }

        if yes_price is not None:
            data["yes_price"] = yes_price
        if no_price is not None:
            data["no_price"] = no_price
        if expiration_ts:
            data["expiration_ts"] = expiration_ts
        if sell_position_floor is not None:
            data["sell_position_floor"] = sell_position_floor
        if buy_max_cost is not None:
            data["buy_max_cost"] = buy_max_cost

        return self._request("POST", "/portfolio/orders", data=data)

    def get_order(self, order_id: str) -> Dict:
        """Get order status."""
        return self._request("GET", f"/portfolio/orders/{order_id}")

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

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
        """Cancel all open orders (optionally for a specific ticker)."""
        data = {}
        if ticker:
            data["ticker"] = ticker

        return self._request("DELETE", "/portfolio/orders", data=data)

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
                         price_cents: int, size: int = 1) -> Dict:
        """
        Place a limit order (simplified interface).

        Args:
            ticker: Market ticker
            side: "buy" or "sell"
            price_cents: Limit price in cents (1-99)
            size: Number of contracts

        Returns:
            Order response with order_id
        """
        # Kalshi orders are on YES/NO markets, need to convert buy/sell to yes/no + action
        # For market making, we typically:
        # - Buy YES at price X = same as Sell NO at price (100-X)
        # - Sell YES at price X = same as Buy NO at price (100-X)

        # Simplified: assume we're always trading the YES side
        if side == "buy":
            return self.create_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=size,
                type="limit",
                yes_price=price_cents,
            )
        else:  # sell
            return self.create_order(
                ticker=ticker,
                side="yes",
                action="sell",
                count=size,
                type="limit",
                yes_price=price_cents,
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
    """

    def __init__(self, api_key: str, api_secret: str, initial_balance: float = 10000.0):
        super().__init__(api_key, api_secret)
        self.paper_balance = initial_balance
        self.paper_positions: Dict[str, int] = {}
        self.paper_orders: Dict[str, Dict] = {}
        self.paper_fills: List[Dict] = []
        self.order_counter = 0

    def create_order(self, ticker: str, side: str, action: str, count: int, **kwargs) -> Dict:
        """Simulate order creation."""
        self.order_counter += 1
        order_id = f"paper_order_{self.order_counter}"

        order = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "status": "resting",
            "created_time": int(time.time() * 1000),
            **kwargs
        }

        self.paper_orders[order_id] = order

        print(f"[PAPER] Order created: {order_id} - {action} {count} {ticker} @ {kwargs.get('yes_price', 'market')}")

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
            print(f"[PAPER] Order canceled: {order_id}")
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

        print(f"[PAPER] Fill executed: {action} {fill_count} {ticker} @ {fill_price}¢")

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
