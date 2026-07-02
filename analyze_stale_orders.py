"""Analyze open orders to find stale ones (far from current market price)."""
import os
from datetime import datetime
from dotenv import load_dotenv
from rl_bot.kalshi_api import KalshiRESTClient

load_dotenv()

# Initialize API client
client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

# Login
email = os.getenv("KALSHI_EMAIL")
password = os.getenv("KALSHI_PASSWORD")
if email and password:
    client.login(email, password)

print("=" * 80)
print("STALE ORDER ANALYSIS")
print("=" * 80)
print()

# Get all open orders
print("Fetching open orders...")
orders_response = client.get_orders(status="resting", limit=1000)
orders = orders_response.get("orders", [])
print(f"Total open orders: {len(orders)}")
print()

# Group orders by ticker
by_ticker = {}
for order in orders:
    ticker = order.get("ticker")  # Changed from "market_ticker" to "ticker"
    if ticker and ticker != "Unknown":
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(order)

print(f"Markets with open orders: {len(by_ticker)}")
print()

# Analyze each ticker
stale_orders = []
good_orders = []
market_data = {}

print("Analyzing order prices vs current market...")
print()

for ticker, ticker_orders in sorted(by_ticker.items()):
    # Get current market data
    try:
        market = client.get_market(ticker)

        # Extract current best bid/ask
        yes_bid = float(market.get("yes_bid", 0))
        yes_ask = float(market.get("yes_ask", 1))
        no_bid = float(market.get("no_bid", 0))
        no_ask = float(market.get("no_ask", 1))

        market_data[ticker] = {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "no_bid": no_bid,
            "no_ask": no_ask,
        }

        # Check each order for staleness
        for order in ticker_orders:
            side = order.get("side")  # "yes" or "no"
            action = order.get("action")  # "buy" or "sell"
            order_price = float(order.get(f"{side}_price_dollars", 0))

            # Define staleness criteria
            # For buy orders: stale if > X cents above current ask
            # For sell orders: stale if > X cents below current bid
            staleness_threshold = 0.10  # 10 cents

            is_stale = False
            reason = ""

            if action == "buy":
                # Buy order - compare to current ask
                current_ask = yes_ask if side == "yes" else no_ask
                if order_price < current_ask - staleness_threshold:
                    is_stale = True
                    reason = f"Buy @ ${order_price:.2f}, market ask ${current_ask:.2f} (+{current_ask - order_price:.2f})"

            else:  # sell
                # Sell order - compare to current bid
                current_bid = yes_bid if side == "yes" else no_bid
                if order_price > current_bid + staleness_threshold:
                    is_stale = True
                    reason = f"Sell @ ${order_price:.2f}, market bid ${current_bid:.2f} (-{order_price - current_bid:.2f})"

            if is_stale:
                stale_orders.append({
                    "order": order,
                    "ticker": ticker,
                    "reason": reason,
                    "distance": abs(order_price - (current_ask if action == "buy" else current_bid))
                })
            else:
                good_orders.append(order)

    except Exception as e:
        print(f"Warning: Could not fetch market data for {ticker}: {e}")

print(f"Analysis complete:")
print(f"  Good orders (near market): {len(good_orders)}")
print(f"  Stale orders (far from market): {len(stale_orders)}")
print()

if stale_orders:
    # Sort by staleness (most stale first)
    stale_orders.sort(key=lambda x: x["distance"], reverse=True)

    print("=" * 80)
    print("STALE ORDERS (Top 20 most stale)")
    print("=" * 80)
    print()

    for i, item in enumerate(stale_orders[:20], 1):
        order = item["order"]
        ticker = item["ticker"]
        reason = item["reason"]
        order_id = order.get("order_id")

        print(f"{i}. {ticker}")
        print(f"   Order ID: {order_id}")
        print(f"   {reason}")
        print()

    if len(stale_orders) > 20:
        print(f"... and {len(stale_orders) - 20} more stale orders")
        print()

    # Calculate capital freed
    estimated_capital_freed = len(stale_orders) * 0.50  # Rough estimate

    print("=" * 80)
    print(f"RECOMMENDATION:")
    print(f"  Cancel {len(stale_orders)} stale orders")
    print(f"  Estimated capital freed: ${estimated_capital_freed:.2f}")
    print(f"  Keep {len(good_orders)} orders near current market prices")
    print()

    # Ask for confirmation
    print("=" * 80)
    print("Would you like to cancel these stale orders? (y/n)")

else:
    print("No stale orders found - all orders are reasonably close to current market prices!")
