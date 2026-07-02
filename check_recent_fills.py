"""Check recent fills from the bot."""
import os
import time
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
print("RECENT FILLS CHECK")
print("=" * 80)
print()

# Get fills from last 30 minutes
now = int(time.time() * 1000)
thirty_min_ago = int((time.time() - 30 * 60) * 1000)

print(f"Checking fills from last 30 minutes...")
print(f"Time range: {datetime.fromtimestamp(thirty_min_ago/1000)} to {datetime.fromtimestamp(now/1000)}")
print()

fills_response = client.get_fills(min_ts=thirty_min_ago, max_ts=now, limit=500)
fills = fills_response.get("fills", [])

if fills:
    print(f"✅ Found {len(fills)} fills in last 30 minutes")
    print()

    # Group by category
    by_category = {}
    total_value = 0
    buy_count = 0
    sell_count = 0

    for fill in fills:
        ticker = fill.get("market_ticker", "Unknown")
        category = ticker.split("-")[0] if "-" in ticker else ticker
        action = fill.get("action", "unknown")
        side = fill.get("side", "unknown")
        count = float(fill.get("count_fp", 0))

        if side == "yes":
            price = float(fill.get("yes_price_dollars", 0))
        else:
            price = float(fill.get("no_price_dollars", 0))

        value = price * count

        if action == "buy":
            buy_count += count
            total_value -= value  # Cost
        else:
            sell_count += count
            total_value += value  # Revenue

        if category not in by_category:
            by_category[category] = {"fills": [], "count": 0, "value": 0}

        by_category[category]["fills"].append(fill)
        by_category[category]["count"] += count
        if action == "buy":
            by_category[category]["value"] -= value
        else:
            by_category[category]["value"] += value

    print("FILLS BY CATEGORY:")
    print("-" * 80)
    for category in sorted(by_category.keys(), key=lambda x: by_category[x]["count"], reverse=True):
        stats = by_category[category]
        print(f"{category}: {len(stats['fills'])} fills, {stats['count']:.0f} contracts, ${stats['value']:+.2f} net")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total fills: {len(fills)}")
    print(f"Buy fills: {buy_count:.0f} contracts")
    print(f"Sell fills: {sell_count:.0f} contracts")
    print(f"Net cash flow: ${total_value:+.2f}")
    print()

    # Show last 20 fills
    print("LAST 20 FILLS:")
    print("-" * 80)
    for fill in fills[-20:]:
        ticker = fill.get("market_ticker", "Unknown")
        action = fill.get("action", "unknown")
        side = fill.get("side", "unknown")
        count = float(fill.get("count_fp", 0))

        if side == "yes":
            price = float(fill.get("yes_price_dollars", 0))
        else:
            price = float(fill.get("no_price_dollars", 0))

        ts = fill.get("created_time", "")
        # Parse timestamp
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            time_str = dt.strftime("%H:%M:%S")
        except:
            time_str = ts

        print(f"{time_str} | {ticker:<45} | {action:>4} {side} x{count:.0f} @ ${price:.2f}")

else:
    print("❌ No fills found in last 30 minutes")
    print()
    print("Bot may be:")
    print("  - Still loading and subscribing to markets")
    print("  - Waiting for profitable opportunities")
    print("  - Experiencing low market activity")

print()
print("=" * 80)
