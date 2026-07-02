"""Fetch all open markets from Kalshi to prepare for full deployment."""
from rl_bot.kalshi_api import KalshiRESTClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

print("Fetching all open markets from Kalshi...")
print("=" * 80)

all_markets = []
cursor = None
page = 1

while True:
    print(f"Fetching page {page}...")

    # Get markets with pagination
    params = {
        "status": "open",
        "limit": 200,  # Max per page
    }
    if cursor:
        params["cursor"] = cursor

    try:
        response = client.get_markets(**params)
        markets = response.get("markets", [])
        cursor = response.get("cursor")

        print(f"  Found {len(markets)} markets on page {page}")
        all_markets.extend(markets)

        # Break if no more pages
        if not cursor or len(markets) == 0:
            break

        page += 1

    except Exception as e:
        print(f"Error fetching markets: {e}")
        break

print("=" * 80)
print(f"Total open markets found: {len(all_markets)}")
print()

# Group by series ticker
series_groups = {}
for market in all_markets:
    series = market.get("event_ticker", "UNKNOWN")
    if series not in series_groups:
        series_groups[series] = []
    series_groups[series].append(market)

print(f"Unique series (categories): {len(series_groups)}")
print()

# Show series breakdown
print("Series breakdown:")
print("-" * 80)
for series, markets in sorted(series_groups.items(), key=lambda x: -len(x[1]))[:50]:
    print(f"  {series}: {len(markets)} markets")

# Save to file for analysis
output = {
    "total_markets": len(all_markets),
    "total_series": len(series_groups),
    "markets": all_markets,
    "series_breakdown": {series: len(markets) for series, markets in series_groups.items()}
}

with open("all_open_markets.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("Saved detailed market data to: all_open_markets.json")
