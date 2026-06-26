"""
Fetch last 2 weeks of Kalshi BTC markets via REST API.

Uses authenticated GET requests to retrieve historical market data
for Bitcoin prediction markets from Kalshi's public API.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import polars as pl
import requests

from authentication_to_kalshi.auth import load_private_key, sign_pss
from config import API_KEY, KEY_PATH

# Kalshi REST API base URL
BASE_URL = "https://api.elections.kalshi.com"

# Alternative: production API
# BASE_URL = "https://trading-api.kalshi.com"


def make_authenticated_request(
    method: str,
    path: str,
    params: Optional[Dict] = None
) -> Dict:
    """
    Make authenticated request to Kalshi REST API.

    Kalshi requires RSA-PSS signed requests with:
    - API key in header
    - Timestamp in milliseconds
    - Signature of: timestamp + method + path

    Args:
        method: HTTP method (GET, POST, etc.)
        path: API endpoint path (e.g., "/trade-api/v2/events")
        params: Optional query parameters

    Returns:
        JSON response from API
    """
    # Load private key for signing
    private_key = load_private_key(KEY_PATH)

    # Current time in milliseconds
    timestamp_ms = str(int(time.time() * 1000))

    # Construct message to sign: timestamp + method + path
    # Note: path only, not full URL, no query params in signature
    message = timestamp_ms + method.upper() + path

    # Sign the message
    signature = sign_pss(private_key, message)

    # Build headers
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "Content-Type": "application/json",
    }

    # Make request
    url = BASE_URL + path
    response = requests.request(
        method,
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    # Raise for HTTP errors
    response.raise_for_status()

    return response.json()


def fetch_btc_markets(days_back: int = 14) -> pl.DataFrame:
    """
    Fetch all BTC markets from the last N days.

    Queries Kalshi's /markets endpoint filtered by:
    - Series ticker contains "BTC"
    - Close time within last N days

    Args:
        days_back: Number of days to look back (default 14)

    Returns:
        Polars DataFrame with market data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Fetching BTC markets from {start_date.date()} to {end_date.date()}...")

    # Fetch markets from API
    # API endpoint: GET /trade-api/v2/markets
    # Params: series_ticker, limit, cursor for pagination
    path = "/trade-api/v2/markets"

    all_markets = []
    cursor = None
    page = 1

    while True:
        print(f"Fetching page {page}...")

        # Query params
        params = {
            "series_ticker": "KXBTC",  # BTC market prefix
            "limit": 200,  # Max results per page
        }

        if cursor:
            params["cursor"] = cursor

        # Make API request
        response = make_authenticated_request("GET", path, params)

        # Extract markets
        markets = response.get("markets", [])
        if not markets:
            break

        print(f"  Retrieved {len(markets)} markets")

        # Filter by date range (close_time)
        for market in markets:
            # Parse close_time (ISO 8601 format)
            close_time_str = market.get("close_time")
            if close_time_str:
                # Convert to datetime (handle timezone)
                close_time = datetime.fromisoformat(
                    close_time_str.replace('Z', '+00:00')
                )

                # Check if within date range
                if start_date <= close_time <= end_date:
                    all_markets.append(market)

        # Check for next page
        cursor = response.get("cursor")
        if not cursor:
            break

        page += 1

    print(f"\nTotal markets found: {len(all_markets)}")

    # Convert to Polars DataFrame
    if not all_markets:
        print("No markets found in date range")
        return pl.DataFrame()

    # Extract relevant fields
    # Market structure: ticker, title, open_time, close_time,
    # yes_bid, yes_ask, last_price, volume, etc.
    records = []
    for m in all_markets:
        records.append({
            "ticker": m.get("ticker"),
            "title": m.get("title"),
            "subtitle": m.get("subtitle"),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
            "status": m.get("status"),
            "yes_bid": m.get("yes_bid"),
            "yes_ask": m.get("yes_ask"),
            "last_price": m.get("last_price"),
            "volume": m.get("volume"),
            "open_interest": m.get("open_interest"),
            "liquidity": m.get("liquidity"),
            "result": m.get("result"),
        })

    df = pl.DataFrame(records)

    # Convert price fields from cents to decimals (0-1 range)
    # Kalshi API returns prices in cents (0-100)
    price_cols = ["yes_bid", "yes_ask", "last_price"]
    for col in price_cols:
        if col in df.columns:
            df = df.with_columns(
                (pl.col(col) / 100.0).alias(col)
            )

    # Sort by close_time descending (newest first)
    df = df.sort("close_time", descending=True)

    return df


def main():
    """
    Fetch and display last 2 weeks of BTC markets.
    """
    # Check credentials
    if not API_KEY:
        print("ERROR: PROD_API_KEY not found in .env file")
        print("\nPlease create a .env file with your Kalshi API credentials:")
        print("  PROD_API_KEY=your_api_key_here")
        print("  PROD_KEY_PATH=path/to/your/private_key.pem")
        return

    if not KEY_PATH:
        print("ERROR: PROD_KEY_PATH not found in .env file")
        return

    # Fetch markets
    df = fetch_btc_markets(days_back=14)

    if df.is_empty():
        print("No markets found")
        return

    # Display summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total markets: {len(df)}")
    print(f"Status breakdown:")
    print(df.group_by("status").agg(pl.count()).sort("count", descending=True))

    print(f"\n{'='*80}")
    print("SAMPLE MARKETS (first 10)")
    print(f"{'='*80}")
    # Show first 10 with key columns
    display_cols = [
        "ticker", "title", "close_time", "status",
        "last_price", "volume", "result"
    ]
    print(df.select(display_cols).head(10))

    # Save to CSV
    output_path = "output/btc_markets_2weeks.csv"
    df.write_csv(output_path)
    print(f"\n✓ Saved to {output_path}")

    # Save to parquet for faster loading
    parquet_path = "output/btc_markets_2weeks.parquet"
    df.write_parquet(parquet_path)
    print(f"✓ Saved to {parquet_path}")


if __name__ == "__main__":
    main()