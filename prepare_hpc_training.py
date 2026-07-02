"""
Download S3 collector data, merge hourly files, and generate per-category
SLURM jobs for HPC training.

Usage:
    python prepare_hpc_training.py                    # all data in S3
    python prepare_hpc_training.py --min-trades 50    # skip low-activity categories
    python prepare_hpc_training.py --dry-run           # show categories without writing
"""

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


S3_BUCKET = "kalshi-data-prod"
OUTPUT_DIR = Path("output")
SLURM_DIR = Path("hpc/mm_jobs")
HPC_PROJECT_DIR = "/scratch/mtk9va/kalshi_v5"

# SLURM template for one category
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J mm_{category}
#SBATCH -o hpc/mm_results/mm_{category}-%A.out
#SBATCH -e hpc/mm_results/mm_{category}-%A.err
#SBATCH -p standard
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -A sds_capstone_atashman

# Train MM PPO agent for {category} category
# Uses 8 CPU cores, 16GB RAM, 4h wall time (standard partition — no GPU wait)

set -euo pipefail

PROJECT_DIR="{hpc_dir}"
cd "$PROJECT_DIR"

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

# Force CPU device (no GPU on standard partition)
export CUDA_VISIBLE_DEVICES=""

# Train MM PPO agent for this category
echo ""
echo "=== Starting MM training for {category} ==="
python -m rl_bot.mm_train \\
    --data output/collector_trades.parquet \\
    --orderbooks output/collector_orderbooks.parquet \\
    --category {category} \\
    --timesteps {timesteps} \\
    --ppo-epochs 2 \\
    --split-date {split_datetime} \\
    --split-mode train \\
    --run-name mm_{category}

# Evaluate on held-out test set (last ~2 hours)
echo ""
echo "=== Evaluating on test split ==="
python -m rl_bot.mm_train \\
    --data output/collector_trades.parquet \\
    --orderbooks output/collector_orderbooks.parquet \\
    --category {category} \\
    --timesteps 1 \\
    --split-date {split_datetime} \\
    --split-mode test \\
    --run-name mm_{category}_test

echo ""
echo "=== Training + eval complete for {category} ==="
echo "Model: $PROJECT_DIR/rl_bot/mm_checkpoints/mm_{category}_{category}_final.zip"
"""


def download_s3_data() -> tuple[Path, Path]:
    """Download all trades and orderbooks from S3 into output/."""
    trades_dir = OUTPUT_DIR / "s3_trades"
    ob_dir = OUTPUT_DIR / "s3_orderbooks"
    trades_dir.mkdir(parents=True, exist_ok=True)
    ob_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading trades from S3...")
    subprocess.run(
        ["aws", "s3", "sync", f"s3://{S3_BUCKET}/trades/", str(trades_dir)],
        check=True,
    )

    print("Downloading orderbooks from S3...")
    subprocess.run(
        ["aws", "s3", "sync", f"s3://{S3_BUCKET}/orderbooks/", str(ob_dir)],
        check=True,
    )

    return trades_dir, ob_dir


def merge_parquet_dir(directory: Path, output_path: Path, label: str) -> pl.DataFrame:
    """Read all parquet files under a directory tree and merge into one file."""
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {directory}")

    print(f"Merging {len(files)} {label} files...")
    dfs = [pl.read_parquet(f) for f in files]
    merged = pl.concat(dfs)

    # Deduplicate trades by (ticker, ts) to handle overlapping flush windows
    if "ts" in merged.columns and "ticker" in merged.columns:
        before = len(merged)
        merged = merged.unique(subset=["ticker", "ts"])
        after = len(merged)
        if before != after:
            print(f"  Deduplicated: {before:,} → {after:,} rows")

    merged.write_parquet(output_path)
    print(f"  Saved {len(merged):,} rows to {output_path}")
    return merged


def extract_category(ticker: str) -> str:
    """Extract category prefix from a ticker.

    Examples:
        KXBTC15M-26JUN291800-00  → KXBTC15M
        KXAFCCLGAME-24JUN29-B10  → KXAFCCLGAME
        FEDHIKE-26JUL             → FEDHIKE
    """
    # Category is everything before the first hyphen
    return ticker.split("-")[0]


def generate_slurm_jobs(
    trades_df: pl.DataFrame,
    min_trades: int,
    timesteps: int,
    split_datetime: str,
    dry_run: bool,
) -> list[str]:
    """Generate a SLURM .slurm file per category that has enough trades."""
    # Count trades per category
    cats = (
        trades_df
        .with_columns(
            pl.col("ticker").map_elements(extract_category, return_dtype=pl.Utf8).alias("category")
        )
        .group_by("category")
        .agg(
            pl.len().alias("trade_count"),
            pl.col("ticker").n_unique().alias("ticker_count"),
        )
        .sort("trade_count", descending=True)
    )

    # Filter to categories with enough data
    cats = cats.filter(pl.col("trade_count") >= min_trades)
    categories = cats["category"].to_list()

    print(f"\n{len(categories)} categories with >= {min_trades} trades:")
    print(cats.head(20))

    if dry_run:
        print("\n[DRY RUN] No slurm files written.")
        return categories

    # Write slurm files
    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        slurm_content = SLURM_TEMPLATE.format(
            category=cat,
            hpc_dir=HPC_PROJECT_DIR,
            timesteps=timesteps,
            split_datetime=split_datetime,
        )
        slurm_path = SLURM_DIR / f"mm_{cat}.slurm"
        slurm_path.write_text(slurm_content)

    print(f"\nWrote {len(categories)} slurm files to {SLURM_DIR}/")

    # Write a batch submission script
    submit_path = Path("hpc/submit_all_mm.sh")
    lines = ["#!/bin/bash", "# Submit all MM training jobs", "set -euo pipefail", ""]
    for cat in categories:
        lines.append(f"sbatch hpc/mm_jobs/mm_{cat}.slurm")
    lines.append("")
    lines.append(f'echo "Submitted {len(categories)} jobs"')
    submit_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote batch submission script: {submit_path}")

    return categories


def main():
    parser = argparse.ArgumentParser(description="Prepare S3 data for HPC training")
    parser.add_argument("--min-trades", type=int, default=20,
                        help="Minimum trades per category to generate a job (default: 20)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Training timesteps per category (default: 500000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show categories without writing slurm files")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip S3 download (use existing local files)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download from S3
    if not args.skip_download:
        trades_dir, ob_dir = download_s3_data()
    else:
        trades_dir = OUTPUT_DIR / "s3_trades"
        ob_dir = OUTPUT_DIR / "s3_orderbooks"

    # 2. Merge hourly parquet files
    trades_path = OUTPUT_DIR / "collector_trades.parquet"
    ob_path = OUTPUT_DIR / "collector_orderbooks.parquet"

    trades_df = merge_parquet_dir(trades_dir, trades_path, "trades")
    merge_parquet_dir(ob_dir, ob_path, "orderbooks")

    # 3. Summary
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Trades:     {len(trades_df):,} rows")
    print(f"Tickers:    {trades_df['ticker'].n_unique():,}")
    print(f"Categories: {trades_df.with_columns(pl.col('ticker').map_elements(extract_category, return_dtype=pl.Utf8).alias('cat'))['cat'].n_unique()}")
    print()

    # 4. Compute temporal split: train on all but last 2 hours, test on last 2 hours
    max_ts = trades_df["ts"].max()
    split_ts = max_ts - 2 * 3600 * 1000  # 2 hours before end (ts is in ms)
    split_dt = datetime.fromtimestamp(split_ts / 1000, tz=timezone.utc)
    split_datetime = split_dt.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"Temporal split: train < {split_datetime} UTC, test >= {split_datetime} UTC")
    train_count = len(trades_df.filter(pl.col("ts") < split_ts))
    test_count = len(trades_df.filter(pl.col("ts") >= split_ts))
    print(f"  Train: {train_count:,} trades, Test: {test_count:,} trades")
    print()

    # 5. Generate SLURM jobs
    categories = generate_slurm_jobs(
        trades_df, args.min_trades, args.timesteps, split_datetime, args.dry_run,
    )

    # 6. Print deploy instructions
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("1. Deploy to HPC:")
        print("     bash hpc/deploy.sh")
        print("2. SSH to HPC and submit jobs:")
        print("     ssh mtk9va@login.hpc.virginia.edu")
        print(f"     cd {HPC_PROJECT_DIR}")
        print("     bash hpc/submit_all_mm.sh")
        print("3. Monitor:")
        print("     squeue -u mtk9va")


if __name__ == "__main__":
    main()
