"""Fetch last 24 hours of trades from Kalshi."""
import os
import time
from datetime import datetime, timedelta
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

# Calculate 24 hours ago in milliseconds
now = int(time.time() * 1000)
twenty_four_hours_ago = int((time.time() - 24 * 60 * 60) * 1000)

print("=" * 80)
print("FETCHING LAST 24 HOURS OF TRADES")
print("=" * 80)
print(f"From: {datetime.fromtimestamp(twenty_four_hours_ago/1000)}")
print(f"To:   {datetime.fromtimestamp(now/1000)}")
print()

# Fetch fills from last 24 hours (max 1000)
print("Fetching fills...", end=" ")

response = client.get_fills(
    min_ts=twenty_four_hours_ago,
    max_ts=now,
    limit=1000
)

all_fills = response.get("fills", [])
print(f"Got {len(all_fills)} fills")

print()
print("=" * 80)
print(f"TOTAL FILLS: {len(all_fills)}")
print("=" * 80)
print()

# Debug: print first fill to see structure
if all_fills:
    print("SAMPLE FILL STRUCTURE:")
    print("-" * 80)
    import json
    print(json.dumps(all_fills[0], indent=2))
    print()

# Analyze fills
if all_fills:
    # Group by ticker
    by_ticker = {}
    total_pnl = 0.0
    buy_count = 0
    sell_count = 0

    for fill in all_fills:
        ticker = fill.get("market_ticker", "Unknown")
        action = fill.get("action", "unknown")  # "buy" or "sell"
        side = fill.get("side", "unknown")  # "yes" or "no"

        # Get the price based on which side was filled
        if side == "yes":
            price = float(fill.get("yes_price_dollars", 0))
        else:
            price = float(fill.get("no_price_dollars", 0))

        count = float(fill.get("count_fp", 1))

        if ticker not in by_ticker:
            by_ticker[ticker] = {"buys": 0, "sells": 0, "volume": 0, "value": 0}

        if action == "buy":
            buy_count += count
            by_ticker[ticker]["buys"] += count
            by_ticker[ticker]["value"] -= price * count  # Cost of buying
        else:
            sell_count += count
            by_ticker[ticker]["sells"] += count
            by_ticker[ticker]["value"] += price * count  # Revenue from selling

        by_ticker[ticker]["volume"] += count

    # Print summary
    print(f"Total Trades: {len(all_fills)}")
    print(f"Buy Orders:  {buy_count}")
    print(f"Sell Orders: {sell_count}")
    print()

    # Top 10 most traded tickers
    print("TOP 10 MOST TRADED TICKERS:")
    print("-" * 80)
    sorted_tickers = sorted(by_ticker.items(), key=lambda x: x[1]["volume"], reverse=True)[:10]
    for ticker, stats in sorted_tickers:
        net_position = stats["buys"] - stats["sells"]
        print(f"{ticker:<50} Vol: {stats['volume']:>3} | Net: {net_position:>+3} | Value: ${stats['value']:>6.2f}")
    print()

    # Recent fills
    print("LAST 20 FILLS:")
    print("-" * 80)
    for fill in all_fills[-20:]:
        ticker = fill.get("market_ticker", "Unknown")
        action = fill.get("action", "unknown")
        side = fill.get("side", "unknown")

        # Get the price based on which side was filled
        if side == "yes":
            price = float(fill.get("yes_price_dollars", 0))
        else:
            price = float(fill.get("no_price_dollars", 0))

        count = float(fill.get("count_fp", 1))
        ts = fill.get("created_time", "")

        print(f"{ts} | {ticker:<40} | {action:>4} {side} {count:.0f}x @ ${price:.2f}")

    print()
    print("=" * 80)

    # Calculate net cash flow
    total_cash_flow = sum(stats["value"] for stats in by_ticker.values())
    print(f"Net Cash Flow (last 24h): ${total_cash_flow:.2f}")
    print("(Negative = net buying, Positive = net selling)")
    print()

else:
    print("No fills in the last 24 hours")
