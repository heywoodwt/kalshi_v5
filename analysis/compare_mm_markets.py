"""Market-Making Market Category Ranking Analysis.

Analyzes MM training results across market categories to identify the most
profitable markets. Reads CSV logs produced by mm_train.py and computes PnL,
fill rates, spreads, and inventory metrics.

Usage:
    # Analyze all CSVs in default output directory
    python -m analysis.compare_mm_markets

    # Specify custom directory
    python -m analysis.compare_mm_markets --output-dir rl_bot/mm_logs/my_run

Key metrics computed:
  - total_pnl: cumulative profit/loss from last row
  - pnl_per_step: average PnL per trading step
  - total_fills: count of steps with buy or sell fills
  - fill_rate: percentage of steps with fills
  - avg_spread_captured: mean bid-ask spread on filled trades
  - avg_inventory: mean absolute inventory level

Time complexity: O(n) where n = total rows across all CSVs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_category_from_filename(csv_path: Path) -> str:
    """Extract market category from CSV filename.

    Expected patterns:
      - output/mm_trades_mm_KXBTC15M.csv -> KXBTC15M
      - output/mm_trades_INX.csv -> INX
      - rl_bot/mm_logs/run1/KXBTC.csv -> KXBTC

    Args:
        csv_path: Path to CSV file

    Returns:
        Category string extracted from filename

    Time complexity: O(1)
    """
    stem = csv_path.stem  # filename without extension

    # Handle pattern: mm_trades_mm_CATEGORY or mm_trades_CATEGORY
    if stem.startswith("mm_trades_"):
        # Remove "mm_trades_" prefix
        category = stem.replace("mm_trades_", "")
        # Remove "mm_" prefix if present (e.g., mm_KXBTC15M -> KXBTC15M)
        if category.startswith("mm_"):
            category = category[3:]
        return category

    # Otherwise use stem as-is (e.g., KXBTC.csv -> KXBTC)
    return stem


def compute_metrics(df: pl.DataFrame, category: str) -> dict[str, any]:
    """Compute market-making metrics for a single category.

    Args:
        df: DataFrame with columns: step, ticker, bid, ask, inventory,
            fills_buy, fills_sell, pnl, reward, half_spread, skew
        category: Market category name

    Returns:
        Dictionary with computed metrics

    Time complexity: O(n) where n = len(df)
    """
    # Basic validation
    if len(df) == 0:
        return {
            "category": category,
            "total_pnl": 0.0,
            "pnl_per_step": 0.0,
            "total_fills": 0,
            "fill_rate": 0.0,
            "avg_spread_captured": 0.0,
            "avg_inventory": 0.0,
            "steps": 0,
        }

    # Total PnL: last row's pnl value
    total_pnl = df["pnl"][-1]

    # PnL per step
    steps = len(df)
    pnl_per_step = total_pnl / steps if steps > 0 else 0.0

    # Identify rows with fills (fills_buy + fills_sell > 0)
    filled_rows = df.filter(
        (pl.col("fills_buy") + pl.col("fills_sell")) > 0
    )
    total_fills = len(filled_rows)
    fill_rate = total_fills / steps if steps > 0 else 0.0

    # Average spread captured on filled trades
    # Spread = ask - bid
    if total_fills > 0:
        avg_spread_captured = filled_rows.select(
            (pl.col("ask") - pl.col("bid")).mean()
        ).item()
    else:
        avg_spread_captured = 0.0

    # Average absolute inventory
    avg_inventory = df["inventory"].abs().mean()

    return {
        "category": category,
        "total_pnl": total_pnl,
        "pnl_per_step": pnl_per_step,
        "total_fills": total_fills,
        "fill_rate": fill_rate,
        "avg_spread_captured": avg_spread_captured,
        "avg_inventory": avg_inventory,
        "steps": steps,
    }


def analyze_per_ticker(df: pl.DataFrame, category: str) -> None:
    """Print per-ticker breakdown for a category.

    Groups by ticker and computes final PnL, fill count, and average inventory
    for each ticker.

    Args:
        df: DataFrame with ticker, pnl, fills_buy, fills_sell, inventory
        category: Market category name

    Time complexity: O(n) where n = len(df)
    """
    print(f"\n  Per-ticker breakdown for {category}:")

    # Group by ticker and compute metrics
    # For each ticker, we want the last PnL value and aggregate stats
    ticker_summary = (
        df.group_by("ticker")
        .agg([
            # Last PnL value for this ticker
            pl.col("pnl").last().alias("final_pnl"),
            # Count of filled rows
            ((pl.col("fills_buy") + pl.col("fills_sell")) > 0).sum().alias("fills"),
            # Mean absolute inventory
            pl.col("inventory").abs().mean().alias("avg_inventory"),
            # Total steps for this ticker
            pl.len().alias("steps"),
        ])
        .sort("final_pnl", descending=True)
    )

    # Print formatted table
    for row in ticker_summary.iter_rows(named=True):
        ticker = row["ticker"]
        pnl = row["final_pnl"]
        fills = row["fills"]
        inv = row["avg_inventory"]
        steps = row["steps"]
        print(f"    {ticker:40} | PnL: ${pnl:8.2f} | Fills: {fills:4} | Avg Inv: {inv:5.2f} | Steps: {steps:5}")


def main() -> None:
    """Main entry point with argparse CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze MM training results and rank market categories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing mm_trades_*.csv files (default: output/)",
    )

    args = parser.parse_args()

    # Convert to Path object
    output_dir = Path(args.output_dir)

    # Find all CSV files matching patterns
    csv_files = []

    # Pattern 1: output/mm_trades_*.csv
    csv_files.extend(output_dir.glob("mm_trades_*.csv"))

    # Pattern 2: *.csv directly in output_dir
    csv_files.extend(output_dir.glob("*.csv"))

    # Pattern 3: subdirectories with *.csv (for mm_logs structure)
    if output_dir.is_dir():
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                csv_files.extend(subdir.glob("*.csv"))

    # Remove duplicates (in case a file matches multiple patterns)
    csv_files = list(set(csv_files))

    if len(csv_files) == 0:
        print(f"No CSV files found in {output_dir}")
        print("Expected patterns:")
        print("  - {output_dir}/mm_trades_*.csv")
        print("  - {output_dir}/*/*.csv")
        return

    print(f"Found {len(csv_files)} CSV files")
    print()

    # Compute metrics for each category
    results = []
    for csv_path in csv_files:
        category = parse_category_from_filename(csv_path)
        print(f"Processing {category} from {csv_path.name}...")

        try:
            # Load CSV with Polars
            df = pl.read_csv(csv_path)

            # Compute metrics
            metrics = compute_metrics(df, category)
            results.append(metrics)

        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
            continue

    if len(results) == 0:
        print("No valid results to analyze")
        return

    # Build summary DataFrame and sort by total_pnl
    summary_df = pl.DataFrame(results).sort("total_pnl", descending=True)

    # Print ranked summary table
    print("\n" + "=" * 100)
    print("RANKED MARKET CATEGORIES BY TOTAL PNL")
    print("=" * 100)
    print(
        f"{'Category':<15} | {'Total PnL':>12} | {'PnL/Step':>12} | "
        f"{'Fills':>8} | {'Fill Rate':>10} | {'Avg Spread':>12} | {'Avg Inv':>10} | {'Steps':>8}"
    )
    print("-" * 100)

    for row in summary_df.iter_rows(named=True):
        cat = row["category"]
        pnl = row["total_pnl"]
        pnl_step = row["pnl_per_step"]
        fills = row["total_fills"]
        fill_rate = row["fill_rate"] * 100  # Convert to percentage
        spread = row["avg_spread_captured"]
        inv = row["avg_inventory"]
        steps = row["steps"]

        print(
            f"{cat:<15} | ${pnl:11.2f} | ${pnl_step:11.6f} | "
            f"{fills:8} | {fill_rate:9.2f}% | ${spread:11.4f} | {inv:10.2f} | {steps:8}"
        )

    print("=" * 100)

    # Identify top-3 categories by PnL
    top_3 = summary_df.head(3)
    print(f"\n\nTOP 3 MOST PROFITABLE CATEGORIES:")
    print("-" * 50)

    for idx, row in enumerate(top_3.iter_rows(named=True), start=1):
        category = row["category"]
        pnl = row["total_pnl"]
        print(f"{idx}. {category}: ${pnl:.2f}")

    # Print per-ticker breakdown for top-3
    print("\n\nPER-TICKER BREAKDOWN (Top 3 Categories):")
    print("=" * 100)

    for row in top_3.iter_rows(named=True):
        category = row["category"]

        # Find the CSV file for this category
        category_csv = None
        for csv_path in csv_files:
            if parse_category_from_filename(csv_path) == category:
                category_csv = csv_path
                break

        if category_csv:
            try:
                df = pl.read_csv(category_csv)
                analyze_per_ticker(df, category)
            except Exception as e:
                print(f"  Error loading {category}: {e}")

    print("\n" + "=" * 100)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
