"""Debug order structure to understand what fields are available."""
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

# Get all open orders
print("Fetching open orders...")
orders_response = client.get_orders(status="resting", limit=10)  # Just first 10 for debugging
orders = orders_response.get("orders", [])

print(f"Total orders: {len(orders)}")
print()

if orders:
    print("First order structure:")
    print(json.dumps(orders[0], indent=2))
    print()

    print("All ticker values:")
    for i, order in enumerate(orders, 1):
        ticker = order.get("ticker")
        market_ticker = order.get("market_ticker")
        print(f"{i}. ticker={ticker}, market_ticker={market_ticker}")
