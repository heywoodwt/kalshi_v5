"""Analyze what's working well in current portfolio."""
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
print("PORTFOLIO PERFORMANCE ANALYSIS")
print("=" * 80)
print()

# Get current balance
balance = client.get_balance()
portfolio_value = float(balance.get("portfolio_value", 0)) / 100
cash = float(balance.get("balance_dollars", 0))

print(f"Portfolio Value: ${portfolio_value:.2f}")
print(f"Cash: ${cash:.2f}")
print()

# Get all positions
positions_response = client.get_positions()
event_positions = positions_response.get("event_positions", [])

# Filter to positions with non-zero exposure
active_positions = [p for p in event_positions if float(p.get("event_exposure_dollars", 0)) != 0]

print(f"Active Positions: {len(active_positions)}")
print()

if active_positions:
    # Analyze by category
    by_category = {}
    total_exposure = 0
    total_realized_pnl = 0

    for pos in active_positions:
        event_ticker = pos.get("event_ticker", "Unknown")
        exposure = float(pos.get("event_exposure_dollars", 0))
        realized_pnl = float(pos.get("realized_pnl_dollars", 0))
        fees_paid = float(pos.get("fees_paid_dollars", 0))
        total_cost = float(pos.get("total_cost_dollars", 0))
        contracts = float(pos.get("total_cost_shares_fp", 0))

        # Extract category from event ticker (e.g., "KXADP-26JUN" -> "KXADP")
        category = event_ticker.split("-")[0] if "-" in event_ticker else event_ticker

        total_exposure += exposure
        total_realized_pnl += realized_pnl

        if category not in by_category:
            by_category[category] = {
                "events": [],
                "total_exposure": 0,
                "total_realized_pnl": 0,
                "total_fees": 0,
            }

        by_category[category]["events"].append({
            "event_ticker": event_ticker,
            "exposure": exposure,
            "realized_pnl": realized_pnl,
            "fees_paid": fees_paid,
            "contracts": contracts,
            "total_cost": total_cost,
        })
        by_category[category]["total_exposure"] += exposure
        by_category[category]["total_realized_pnl"] += realized_pnl
        by_category[category]["total_fees"] += fees_paid

    # Sort categories by realized P&L (most profitable first)
    sorted_categories = sorted(
        by_category.items(),
        key=lambda x: x[1]["total_realized_pnl"],
        reverse=True
    )

    print("=" * 80)
    print("PERFORMANCE BY CATEGORY")
    print("=" * 80)
    print()

    for category, stats in sorted_categories:
        pnl = stats["total_realized_pnl"]
        exposure = stats["total_exposure"]
        fees = stats["total_fees"]
        net_pnl = pnl - fees
        num_events = len(stats["events"])

        status = "✅ PROFITABLE" if net_pnl > 0 else "❌ LOSING" if net_pnl < 0 else "  BREAKEVEN"

        print(f"{status} {category}")
        print(f"  Events: {num_events}")
        print(f"  Exposure: ${exposure:.2f}")
        print(f"  Realized P&L: ${pnl:+.2f}")
        print(f"  Fees Paid: ${fees:.2f}")
        print(f"  Net P&L: ${net_pnl:+.2f}")
        print()

        # Show individual events if category has multiple
        if num_events > 1:
            for event in stats["events"]:
                event_pnl = event["realized_pnl"]
                print(f"    {event['event_ticker']:<30} Exp: ${event['exposure']:>6.2f}  P&L: ${event_pnl:>+6.2f}  Contracts: {event['contracts']:.0f}")
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Exposure: ${total_exposure:.2f}")
    print(f"Total Realized P&L: ${total_realized_pnl:+.2f}")
    print(f"Total Fees: ${sum(cat['total_fees'] for cat in by_category.values()):.2f}")
    print(f"Net P&L: ${total_realized_pnl - sum(cat['total_fees'] for cat in by_category.values()):+.2f}")
    print()

    # Get recent fills (last 6 hours)
    print("=" * 80)
    print("RECENT TRADING ACTIVITY (Last 6 Hours)")
    print("=" * 80)
    print()

    now = int(time.time() * 1000)
    six_hours_ago = int((time.time() - 6 * 60 * 60) * 1000)

    fills = client.get_fills(min_ts=six_hours_ago, max_ts=now, limit=100)
    recent_fills = fills.get("fills", [])

    if recent_fills:
        print(f"Total fills: {len(recent_fills)}")
        print()

        # Group by ticker
        by_ticker = {}
        for fill in recent_fills:
            ticker = fill.get("market_ticker", "Unknown")
            category = ticker.split("-")[0] if "-" in ticker else ticker

            if category not in by_ticker:
                by_ticker[category] = []
            by_ticker[category].append(fill)

        # Show activity by category
        for category, cat_fills in sorted(by_ticker.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"{category}: {len(cat_fills)} fills")

        print()
        print("Last 10 fills:")
        for fill in recent_fills[-10:]:
            ticker = fill.get("market_ticker", "Unknown")
            action = fill.get("action", "unknown")
            side = fill.get("side", "unknown")
            count = float(fill.get("count_fp", 0))

            if side == "yes":
                price = float(fill.get("yes_price_dollars", 0))
            else:
                price = float(fill.get("no_price_dollars", 0))

            ts = fill.get("created_time", "")
            print(f"  {ts} | {ticker:<40} | {action:>4} {side} x{count:.0f} @ ${price:.2f}")
    else:
        print("No fills in last 6 hours")

    print()

else:
    print("No active positions")

print("=" * 80)
