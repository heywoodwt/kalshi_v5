"""Count total markets for all validated categories."""
from pathlib import Path
from rl_bot.kalshi_api import KalshiRESTClient
import os
from dotenv import load_dotenv
import time

load_dotenv()

client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

# Load validated categories
with open("validated_categories_with_markets.txt") as f:
    categories = [line.strip() for line in f if line.strip()]

print(f"Checking {len(categories)} validated categories for open markets...")
print("=" * 80)

categories_with_markets = []
total_markets = 0

for i, category in enumerate(categories, 1):
    try:
        response = client.get_markets(
            series_ticker=category,
            status="open",
            limit=200
        )
        markets = response.get("markets", [])
        market_count = len(markets)

        if market_count > 0:
            categories_with_markets.append(category)
            total_markets += market_count
            print(f"{i:3d}. {category:<30} {market_count:>4} markets")

        # Rate limit: small delay
        if i % 10 == 0:
            time.sleep(0.5)

    except Exception as e:
        print(f"{i:3d}. {category:<30} ERROR: {e}")
        time.sleep(1)

print("=" * 80)
print(f"\nCategories with open markets: {len(categories_with_markets)}")
print(f"Total markets: {total_markets}")
print(f"\nThis is what we can deploy with the current trained models!")
