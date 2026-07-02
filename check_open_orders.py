"""Check current open orders."""
import os
from datetime import datetime
from dotenv import load_dotenv
from rl_bot.kalshi_api import KalshiRESTClient

load_dotenv()

client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

email = os.getenv("KALSHI_EMAIL")
password = os.getenv("KALSHI_PASSWORD")
if email and password:
    client.login(email, password)

# Check current open orders
print("=" * 80)
print("CURRENT OPEN ORDERS")
print("=" * 80)
print()

orders = client.get_orders(status="resting", limit=100)
open_orders = orders.get("orders", [])

print(f"Total open orders: {len(open_orders)}")
print()

if open_orders:
    # Group by category
    by_category = {}
    for order in open_orders:
        ticker = order.get("ticker", "Unknown")
        category = ticker.split("-")[0] if "-" in ticker else ticker

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(order)

    print(f"Categories with orders: {len(by_category)}")
    print()

    for category in sorted(by_category.keys(), key=lambda x: len(by_category[x]), reverse=True):
        print(f"{category}: {len(by_category[category])} orders")

    print()
    print("Most recent 10 orders:")
    print("-" * 80)
    for order in open_orders[:10]:
        ticker = order.get("ticker", "Unknown")
        side = order.get("side", "unknown")
        action = order.get("action", "unknown")
        created = order.get("created_time", "")

        if side == "yes":
            price = float(order.get("yes_price_dollars", 0))
        else:
            price = float(order.get("no_price_dollars", 0))

        try:
            dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = created[:19] if len(created) > 19 else created

        print(f"{time_str} | {ticker:<40} | {action:>4} {side} @ ${price:.3f}")
else:
    print("❌ No open orders - bot has not placed any orders yet")
    print()
    print("This is normal for a newly started bot. Possible reasons:")
    print("  • Waiting for profitable spread opportunities")
    print("  • Still processing initial market data")
    print("  • Running model predictions to find edges")
    print("  • Orderbook spreads may be too wide (no edge)")

print()
print("=" * 80)
