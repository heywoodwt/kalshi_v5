"""Fetch detailed portfolio information from Kalshi."""
import os
import json
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
print("PORTFOLIO DETAILS")
print("=" * 80)
print()

# Get balance
try:
    balance = client.get_balance()
    print("BALANCE:")
    print(json.dumps(balance, indent=2))
    print()
except Exception as e:
    print(f"Could not fetch balance: {e}")
    print()

# Get positions
try:
    positions = client.get_positions()
    print("POSITIONS:")
    print(json.dumps(positions, indent=2))
    print()
except Exception as e:
    print(f"Could not fetch positions: {e}")
    print()

# Get open orders
try:
    orders = client.get_orders(status="resting", limit=1000)
    print(f"OPEN ORDERS: {len(orders.get('orders', []))}")

    if orders.get("orders"):
        total_order_value = 0
        for order in orders.get("orders", []):
            # Estimate locked capital per order
            # For a buy order, capital is locked at the order price
            # For a sell order, capital is locked in the position being sold
            count = order.get("remaining_count", 1)
            total_order_value += count * 0.50  # Rough estimate

        print(f"Estimated capital locked in orders: ${total_order_value:.2f}")
        print()

        # Show first few orders
        print("SAMPLE ORDERS:")
        for order in orders.get("orders", [])[:10]:
            ticker = order.get("market_ticker", "Unknown")
            side = order.get("side", "unknown")
            action = order.get("action", "unknown")
            count = order.get("remaining_count", 0)
            print(f"  {ticker:<50} {action:>4} {side} x{count}")

        if len(orders.get("orders", [])) > 10:
            print(f"  ... and {len(orders.get('orders', [])) - 10} more")
    print()
except Exception as e:
    print(f"Could not fetch orders: {e}")
    print()

# Get fills from last hour to see recent activity
try:
    import time
    now = int(time.time() * 1000)
    one_hour_ago = int((time.time() - 3600) * 1000)

    fills = client.get_fills(min_ts=one_hour_ago, max_ts=now, limit=100)
    recent_fills = fills.get("fills", [])

    print(f"RECENT FILLS (last hour): {len(recent_fills)}")

    if recent_fills:
        for fill in recent_fills[-10:]:
            ticker = fill.get("market_ticker", "Unknown")
            action = fill.get("action", "unknown")
            side = fill.get("side", "unknown")
            count = fill.get("count_fp", 0)
            ts = fill.get("created_time", "")
            print(f"  {ts} | {ticker:<40} | {action} {side} x{count}")
    print()
except Exception as e:
    print(f"Could not fetch recent fills: {e}")
    print()

print("=" * 80)
