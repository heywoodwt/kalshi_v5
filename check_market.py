from rl_bot.kalshi_api import KalshiRESTClient
import os
from dotenv import load_dotenv

load_dotenv()

client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

# Check the KXLOLTOTALMAPS market that failed
ticker = "KXLOLTOTALMAPS-26JUN300400TLKC-5"
try:
    response = client.get_market(ticker)
    market = response.get("market", {})
    print(f"Market: {ticker}")
    print(f"Status: {market.get('status')}")
    print(f"Close time: {market.get('close_time')}")
    print(f"Can trade: {market.get('can_close_early')}")
    print(f"Full response: {market}")
except Exception as e:
    print(f"Error getting market: {e}")

# Try to place a small test order
print("\nAttempting to place order...")
try:
    response = client.place_limit_order(
        ticker=ticker,
        side="buy",
        price_cents=1,
        size=1
    )
    print(f"Order response: {response}")
except Exception as e:
    print(f"Order error: {e}")
    # Try to get more error details
    import traceback
    traceback.print_exc()
