"""Cancel stale orders that are far from current market prices."""
import os
import time
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
print("CANCELING STALE ORDERS")
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
    ticker = order.get("ticker")
    if ticker and ticker != "Unknown":
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(order)

# Analyze each ticker for staleness
stale_orders = []
good_orders = []
staleness_threshold = 0.10  # 10 cents

print(f"Analyzing {len(by_ticker)} markets...")
print()

for i, (ticker, ticker_orders) in enumerate(by_ticker.items(), 1):
    if i % 10 == 0:
        print(f"  Analyzed {i}/{len(by_ticker)} markets...")

    # Get current market data
    try:
        market = client.get_market(ticker)

        # Extract current best bid/ask
        yes_bid = float(market.get("yes_bid", 0))
        yes_ask = float(market.get("yes_ask", 1))
        no_bid = float(market.get("no_bid", 0))
        no_ask = float(market.get("no_ask", 1))

        # Check each order for staleness
        for order in ticker_orders:
            side = order.get("side")  # "yes" or "no"
            action = order.get("action")  # "buy" or "sell"
            order_price = float(order.get(f"{side}_price_dollars", 0))

            is_stale = False

            if action == "buy":
                # Buy order - compare to current ask
                current_ask = yes_ask if side == "yes" else no_ask
                if order_price < current_ask - staleness_threshold:
                    is_stale = True

            else:  # sell
                # Sell order - compare to current bid
                current_bid = yes_bid if side == "yes" else no_bid
                if order_price > current_bid + staleness_threshold:
                    is_stale = True

            if is_stale:
                stale_orders.append(order)
            else:
                good_orders.append(order)

    except Exception as e:
        # If we can't fetch market data, assume orders are stale
        # (market might be closed or delisted)
        for order in ticker_orders:
            stale_orders.append(order)

print()
print(f"Analysis complete:")
print(f"  Stale orders to cancel: {len(stale_orders)}")
print(f"  Good orders to keep: {len(good_orders)}")
print()

if not stale_orders:
    print("No stale orders found!")
    exit(0)

# Cancel stale orders
print("=" * 80)
print("CANCELING STALE ORDERS")
print("=" * 80)
print()

canceled = 0
errors = 0

for i, order in enumerate(stale_orders, 1):
    order_id = order.get("order_id")
    ticker = order.get("ticker", "Unknown")

    try:
        client.cancel_order(order_id)
        canceled += 1

        if i % 10 == 0:
            print(f"  Canceled {i}/{len(stale_orders)} orders...")

        # Rate limit: don't overwhelm the API
        if i % 50 == 0:
            time.sleep(1)  # 1 second pause every 50 cancellations

    except Exception as e:
        errors += 1
        if errors <= 10:  # Only show first 10 errors
            print(f"  Error canceling {order_id} ({ticker}): {e}")

print()
print("=" * 80)
print("CANCELLATION COMPLETE")
print("=" * 80)
print(f"Successfully canceled: {canceled}")
print(f"Errors: {errors}")
print(f"Estimated capital freed: ${canceled * 0.50:.2f}")
print()

# Refresh balance
balance = client.get_balance()
print(f"New balance: ${float(balance.get('balance_dollars', 0)):.2f}")
print()
