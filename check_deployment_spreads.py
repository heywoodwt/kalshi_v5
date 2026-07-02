"""Check current spreads for the 6 deployment categories."""
import os
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

# Deployment plan categories
DEPLOYMENT_CATEGORIES = [
    "KXACBGAME",      # Sports, lowest variance ($27.40)
    "KXATP",          # Tennis ($30.56)
    "KXATPCHALLENGERMATCH",  # Tennis challenger ($28.75)
    "KXAFCCLGAME",    # Football ($50.38)
    "KXAPFDDH",       # Sports division ($19.67)
    "KXBTCD",         # BTC daily ($13.45)
]

print("=" * 80)
print("DEPLOYMENT CATEGORIES - SPREAD ANALYSIS")
print("=" * 80)
print()

results = []

for category in DEPLOYMENT_CATEGORIES:
    print(f"Checking {category}...")

    try:
        # Get active markets for this category
        response = client.get_markets(
            series_ticker=category,
            status="open",
            limit=200
        )

        markets = response.get("markets", [])

        if not markets:
            print(f"  ❌ No open markets for {category}")
            results.append({
                "category": category,
                "markets": 0,
                "min_spread": None,
                "avg_spread": None,
                "tradeable": 0
            })
            continue

        # Check spreads for all markets in this category
        spreads = []
        tradeable_count = 0
        sample_spreads = []  # Store first 5 spreads for display

        for market in markets[:50]:  # Check first 50 markets
            ticker = market["ticker"]

            try:
                # Get orderbook
                ob_response = client.get_orderbook(ticker)
                orderbook = ob_response.get("orderbook", {})

                yes_bids = orderbook.get("yes", [])
                no_bids = orderbook.get("no", [])

                if not yes_bids or not no_bids:
                    continue

                # Calculate spread
                best_bid = float(yes_bids[0][0])
                best_ask = 1.0 - float(no_bids[0][0])

                if best_ask > best_bid:
                    spread = best_ask - best_bid
                    spreads.append(spread)

                    # Store first 5 for display
                    if len(sample_spreads) < 5:
                        sample_spreads.append((ticker, spread, best_bid, best_ask))

                    # Count markets with spread < 0.30 (tradeable)
                    if spread < 0.30:
                        tradeable_count += 1

            except Exception as e:
                continue

        if spreads:
            min_spread = min(spreads)
            avg_spread = sum(spreads) / len(spreads)
            max_spread = max(spreads)

            print(f"  ✓ {len(markets)} markets found")
            print(f"    Min spread: ${min_spread:.3f}")
            print(f"    Avg spread: ${avg_spread:.3f}")
            print(f"    Max spread: ${max_spread:.3f}")
            print(f"    Tradeable (< $0.30): {tradeable_count}/{len(spreads)} markets")

            if sample_spreads:
                print(f"    Sample markets:")
                for ticker, spread, bid, ask in sample_spreads[:3]:
                    print(f"      {ticker}: ${spread:.3f} (bid=${bid:.3f}, ask=${ask:.3f})")

            results.append({
                "category": category,
                "markets": len(markets),
                "min_spread": min_spread,
                "avg_spread": avg_spread,
                "tradeable": tradeable_count
            })
        else:
            print(f"  ⚠️ No valid orderbooks found")
            results.append({
                "category": category,
                "markets": len(markets),
                "min_spread": None,
                "avg_spread": None,
                "tradeable": 0
            })

    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append({
            "category": category,
            "markets": 0,
            "min_spread": None,
            "avg_spread": None,
            "tradeable": 0
        })

    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

total_tradeable = sum(r["tradeable"] for r in results)
total_markets = sum(r["markets"] for r in results)

print(f"Total markets across 6 categories: {total_markets}")
print(f"Total tradeable markets (spread < $0.30): {total_tradeable}")
print()

if total_tradeable > 0:
    print("✅ TRADEABLE CATEGORIES:")
    for r in results:
        if r["tradeable"] > 0:
            print(f"  {r['category']}: {r['tradeable']} tradeable markets (min spread: ${r['min_spread']:.3f})")
    print()
    print("Recommendation: Deploy on categories with tradeable markets")
else:
    print("❌ NO TRADEABLE MARKETS FOUND")
    print()
    print("Possible reasons:")
    print("  - Markets currently have wide spreads (low liquidity)")
    print("  - Wrong time of day (spreads tighten during active hours)")
    print("  - Seasonal effects (some sports categories out of season)")
    print()
    print("Recommendation: Wait for active trading hours or check different categories")

print()
print("=" * 80)
