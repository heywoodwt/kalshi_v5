"""Download June data from S3, retrain models, and deploy."""
import os
import subprocess
from pathlib import Path
import polars as pl

# Download June trades from S3
print("=" * 80)
print("DOWNLOADING JUNE TRADE DATA FROM S3")
print("=" * 80)
print()

os.makedirs("output/june_trades", exist_ok=True)

# Download all June trade files
subprocess.run([
    "aws", "s3", "sync",
    "s3://kalshi-data-prod/trades/2026-06-29/",
    "output/june_trades/",
    "--region", "us-east-2"
], check=True)

subprocess.run([
    "aws", "s3", "sync",
    "s3://kalshi-data-prod/trades/2026-06-30/",
    "output/june_trades/",
    "--region", "us-east-2"
], check=True)

print("✓ Downloaded June trade data")
print()

# Combine all June parquet files
print("=" * 80)
print("COMBINING JUNE TRADE DATA")
print("=" * 80)
print()

june_files = list(Path("output/june_trades").glob("*.parquet"))
print(f"Found {len(june_files)} hourly files")

dfs = []
for f in sorted(june_files):
    print(f"  Reading {f.name}...")
    dfs.append(pl.read_parquet(f))

combined = pl.concat(dfs)
print(f"✓ Combined {len(combined):,} trades")
print()

# Save combined June data
output_path = "output/rl_kalshi_trades_june.parquet"
combined.write_parquet(output_path)
print(f"✓ Saved to {output_path}")
print(f"  Date range: {combined['created_time'].min()} to {combined['created_time'].max()}")
print()

# Get unique categories with significant volume
print("=" * 80)
print("ANALYZING CATEGORIES IN JUNE DATA")
print("=" * 80)
print()

category_stats = (
    combined
    .with_columns(pl.col("ticker").str.extract(r"^([A-Z]+)", 1).alias("category"))
    .group_by("category")
    .agg([
        pl.len().alias("trade_count"),
        pl.col("ticker").n_unique().alias("unique_tickers")
    ])
    .sort("trade_count", descending=True)
)

print("Top 20 categories by trade volume:")
print(category_stats.head(20))
print()

# Filter to categories with significant volume (>1000 trades)
significant_categories = (
    category_stats
    .filter(pl.col("trade_count") > 1000)
    .select("category")
    .to_series()
    .to_list()
)

print(f"✓ Found {len(significant_categories)} categories with >1000 trades")
print()

# Check which of the Top 50 deployment categories have June data
from rl_bot.live_config_top50 import TOP_50_CATEGORIES

top50_names = [cat.name for cat in TOP_50_CATEGORIES]
top50_with_data = [cat for cat in top50_names if cat in significant_categories]

print(f"Top 50 categories with June data: {len(top50_with_data)}/50")
print("Categories with data:")
for cat in top50_with_data:
    count = category_stats.filter(pl.col("category") == cat).select("trade_count").item()
    print(f"  {cat}: {count:,} trades")
print()

if len(top50_with_data) == 0:
    print("❌ NONE of the Top 50 categories have significant June trade volume!")
    print("   This explains why spreads are so wide - these markets are inactive.")
    print()
    print("Recommendation: Use the most active June categories instead")
    print()

    # Show top active categories
    print("Most active categories in June:")
    for i, row in enumerate(category_stats.head(20).iter_rows(named=True), 1):
        print(f"  {i}. {row['category']}: {row['trade_count']:,} trades, {row['unique_tickers']} markets")
else:
    print(f"✓ {len(top50_with_data)} Top 50 categories active in June")
    print("  Ready to retrain these categories on June data")

print()
print("=" * 80)
print("DATA DOWNLOAD AND ANALYSIS COMPLETE")
print("=" * 80)
