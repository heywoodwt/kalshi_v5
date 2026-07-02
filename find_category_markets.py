"""Find all markets for our trained categories."""
from rl_bot.kalshi_api import KalshiRESTClient
import os
from dotenv import load_dotenv

load_dotenv()

client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

# Our trained categories (series tickers)
trained_categories = [
    "KXLOLTOTALMAPS",
    "KXBBCHARTPOSITIONALBUM",
    "KXRAINLAXM",
    "KXNBAMVP",
    "KXSPOTIFYW",
    "KXAFCCLGAME",
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
]

print("Finding all markets for trained categories...")
print("=" * 80)

total_markets = 0
category_markets = {}

for series in trained_categories:
    print(f"\n{series}:")
    try:
        # Get all open markets for this series
        response = client.get_markets(
            series_ticker=series,
            status="open",
            limit=200
        )

        markets = response.get("markets", [])
        category_markets[series] = markets
        total_markets += len(markets)

        print(f"  Found {len(markets)} open markets")

        # Show first 5 market tickers
        for market in markets[:5]:
            print(f"    - {market['ticker']}")

        if len(markets) > 5:
            print(f"    ... and {len(markets) - 5} more")

    except Exception as e:
        print(f"  Error: {e}")
        category_markets[series] = []

print("\n" + "=" * 80)
print(f"Total markets across all trained categories: {total_markets}")
print()

# Show breakdown
print("Category breakdown:")
for series, markets in category_markets.items():
    print(f"  {series}: {len(markets)} markets")
