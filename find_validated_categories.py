"""Find the 262 validated categories with open markets."""
import re
from pathlib import Path
from rl_bot.kalshi_api import KalshiRESTClient
import os
from dotenv import load_dotenv

load_dotenv()

# Parse temporal split results to get validated categories
results_file = Path("hpc/mm_temporal_split_results.md")
validated_categories = []

print("Parsing temporal split results...")
with open(results_file) as f:
    content = f.read()

    # Extract all category rows from markdown tables
    # Format: | KXCATEGORY | $+1.27 | $+14.14 | 33% | 100% | 0.09x |
    pattern = r'\| (KX[A-Z0-9]+) \|'
    matches = re.findall(pattern, content)

    # Get unique categories
    all_categories = sorted(set(matches))

print(f"Found {len(all_categories)} unique categories in results")

# Check which have trained models
checkpoint_dir = Path("rl_bot/mm_checkpoints")
available_models = {f.stem for f in checkpoint_dir.glob("*.zip")}

print(f"Found {len(available_models)} trained models on disk")

# Find intersection: categories in results with trained models
categories_with_models = sorted(set(all_categories) & available_models)
print(f"Categories in results with trained models: {len(categories_with_models)}")

# Check which have open markets on Kalshi
client = KalshiRESTClient(
    api_key=os.getenv("KALSHI_API_KEY"),
    api_secret=os.getenv("KALSHI_API_SECRET")
)

categories_with_markets = []
total_markets = 0

print("\nChecking for open markets (sampling every 10th category for speed)...")
for i, category in enumerate(categories_with_models):
    # Sample every 10th to speed up
    if i % 10 != 0 and i < len(categories_with_models) - 10:
        continue

    try:
        response = client.get_markets(
            series_ticker=category,
            status="open",
            limit=1
        )
        market_count = len(response.get("markets", []))

        if market_count > 0:
            categories_with_markets.append(category)
            print(f"  ✓ {category}: has markets")

    except Exception as e:
        pass

print(f"\nSampled {len(categories_with_markets)} categories with open markets")
print(f"\nEstimated total categories with both models and open markets: ~{len(categories_with_markets) * 10}")

# Save the list
output_file = Path("validated_categories_with_markets.txt")
with open(output_file, "w") as f:
    for cat in categories_with_models:
        f.write(f"{cat}\n")

print(f"\nSaved {len(categories_with_models)} categories to {output_file}")
print(f"\nTo deploy all validated models, update PHASE1_CATEGORIES in live_config_phase1.py")
