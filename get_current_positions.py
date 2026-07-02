"""Fetch and analyze current open positions from Kalshi."""
import os
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
print("CURRENT OPEN POSITIONS")
print("=" * 80)
print()

# Fetch all positions
response = client.get_positions()
positions = response.get("positions", [])

# Filter to non-zero positions
open_positions = [p for p in positions if p.get("total_traded", 0) != 0]

print(f"Total positions: {len(open_positions)}")
print()

if open_positions:
    # Calculate totals
    total_contracts = sum(abs(p.get("total_traded", 0)) for p in open_positions)
    total_value = sum(abs(p.get("total_traded", 0)) * 0.50 for p in open_positions)  # Assume $0.50 avg value

    print(f"Total contracts held: {total_contracts}")
    print(f"Estimated position value: ${total_value:.2f}")
    print()

    # Group by category
    by_category = {}
    for pos in open_positions:
        ticker = pos.get("market_ticker", "Unknown")
        # Extract category from ticker (e.g., "KXADP-24JUN-T341" -> "KXADP")
        category = ticker.split("-")[0] if "-" in ticker else "Unknown"

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(pos)

    print("POSITIONS BY CATEGORY:")
    print("-" * 80)
    for category, cat_positions in sorted(by_category.items()):
        cat_total = sum(abs(p.get("total_traded", 0)) for p in cat_positions)
        cat_value = cat_total * 0.50
        print(f"\n{category}: {len(cat_positions)} markets, {cat_total} contracts, ${cat_value:.2f} value")
        print()

        # Show individual positions
        for pos in sorted(cat_positions, key=lambda x: abs(x.get("total_traded", 0)), reverse=True):
            ticker = pos.get("market_ticker")
            position = pos.get("total_traded", 0)
            direction = "LONG" if position > 0 else "SHORT"

            print(f"  {ticker:<50} {direction:>5} {abs(position):>3} contracts")

    print()
    print("=" * 80)

    # Get open orders
    print("\nFETCHING OPEN ORDERS...")
    orders_response = client.get_orders(status="resting", limit=1000)
    orders = orders_response.get("orders", [])

    print(f"Total open orders: {len(orders)}")
    print()

    if orders:
        # Analyze orders vs positions
        profit_taking_orders = []
        inventory_building_orders = []

        # Build position map
        position_map = {}
        for pos in open_positions:
            ticker = pos.get("market_ticker")
            position_map[ticker] = pos.get("total_traded", 0)

        for order in orders:
            ticker = order.get("market_ticker")
            side = order.get("side")  # "yes" or "no"
            order_id = order.get("order_id")

            current_position = position_map.get(ticker, 0)

            # Classify order
            is_profit_taking = False
            if current_position > 0 and side == "no":  # Long position, sell order
                is_profit_taking = True
            elif current_position < 0 and side == "yes":  # Short position, buy order
                is_profit_taking = True

            if is_profit_taking:
                profit_taking_orders.append(order)
            else:
                inventory_building_orders.append(order)

        print(f"Profit-taking orders (reducing positions): {len(profit_taking_orders)}")
        print(f"Inventory-building orders (adding positions): {len(inventory_building_orders)}")
        print()

        # Show sample profit-taking orders
        if profit_taking_orders:
            print("SAMPLE PROFIT-TAKING ORDERS:")
            print("-" * 80)
            for order in profit_taking_orders[:10]:
                ticker = order.get("market_ticker", "Unknown")
                side = order.get("side", "unknown")
                position = position_map.get(ticker, 0)
                direction = "LONG→SELL" if position > 0 else "SHORT→BUY"
                print(f"  {ticker:<50} {direction:>12} ({side} order)")
            if len(profit_taking_orders) > 10:
                print(f"  ... and {len(profit_taking_orders) - 10} more")
            print()

else:
    print("No open positions")
