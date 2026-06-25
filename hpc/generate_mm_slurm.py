#!/usr/bin/env python3
"""Generate per-category SLURM scripts for MM PPO training on UVA HPC.

This script reads the trade data parquet, discovers categories with sufficient
trade volume, and generates one SLURM job script per category in hpc/mm_jobs/.

Usage:
    python hpc/generate_mm_slurm.py

Output:
    - One .slurm file per category in hpc/mm_jobs/
    - Summary of categories and jobs generated
"""

import polars as pl
from pathlib import Path

# Min trades threshold from MMConfig
MIN_TRADES_PER_TICKER = 50

# SLURM template for per-category training
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -J mm_{category}
#SBATCH -o hpc/mm_results/mm_{category}-%A.out
#SBATCH -e hpc/mm_results/mm_{category}-%A.err
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -A sds_capstone_atashman

# Train MM PPO agent for {category} category
# Uses 1x A100 GPU, 4 CPU cores, 32GB RAM, 4h wall time

set -euo pipefail

PROJECT_DIR="/scratch/mtk9va/kalshi_v5"
cd "$PROJECT_DIR"

# Load modules and activate conda env
module purge
module load miniforge
eval "$(conda shell.bash hook)"
conda activate kalshi_rl

# Show GPU info
echo "=== GPU Info ==="
python -c "
import torch
print(f'CUDA available: {{torch.cuda.is_available()}}')
print(f'GPU count: {{torch.cuda.device_count()}}')
if torch.cuda.device_count() > 0:
    print(f'GPU 0: {{torch.cuda.get_device_name(0)}}')
"

# Train MM PPO agent for this category
echo ""
echo "=== Starting MM training for {category} ==="
python -m rl_bot.mm_train \\
    --data output/rl_all_trades_3mo.parquet \\
    --category {category} \\
    --timesteps 500000 \\
    --run-name mm_{category}

echo ""
echo "=== Training complete for {category} ==="
echo "Model: $PROJECT_DIR/rl_bot/mm_models/mm_{category}.zip"
"""


def main():
    """Generate SLURM scripts for all eligible categories."""
    # Project root (one level up from hpc/)
    project_root = Path(__file__).parent.parent
    data_path = project_root / "output" / "rl_all_trades_3mo.parquet"

    # Output directories
    jobs_dir = project_root / "hpc" / "mm_jobs"
    results_dir = project_root / "hpc" / "mm_results"

    print("=== MM SLURM Generator ===")
    print(f"Data: {data_path}")
    print(f"Output: {jobs_dir}")
    print("")

    # Ensure data file exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Trade data not found: {data_path}\n"
            f"Run rl_bot/mm_train.py first to generate training data."
        )

    # Create output directories
    jobs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Load data and count trades per ticker
    print("Loading trade data...")
    df = pl.read_parquet(data_path)

    # Extract category from ticker (format: CATEGORY-subcat-...)
    # Example: BTC-23DEC-T45000 → category = BTC
    df = df.with_columns([
        pl.col("ticker").str.split("-").list.first().alias("category")
    ])

    # Count trades per ticker, then trades per category
    ticker_counts = (
        df.group_by("ticker")
        .agg(pl.len().alias("n_trades"))
        .filter(pl.col("n_trades") >= MIN_TRADES_PER_TICKER)
    )

    # Get category for each ticker with enough trades
    ticker_with_category = df.select(["ticker", "category"]).unique()
    eligible_tickers = ticker_counts.join(ticker_with_category, on="ticker")

    # Count categories with at least one eligible ticker
    category_stats = (
        eligible_tickers.group_by("category")
        .agg([
            pl.len().alias("n_tickers"),
            pl.col("n_trades").sum().alias("total_trades")
        ])
        .sort("total_trades", descending=True)
    )

    print(f"Found {len(category_stats)} categories with eligible tickers:")
    print(category_stats)
    print("")

    # Generate one SLURM script per category
    generated = []
    for row in category_stats.iter_rows(named=True):
        category = row["category"]
        n_tickers = row["n_tickers"]
        total_trades = row["total_trades"]

        # Generate SLURM script
        script_content = SLURM_TEMPLATE.format(category=category)
        script_path = jobs_dir / f"mm_{category}.slurm"

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        script_path.chmod(0o755)

        generated.append((category, n_tickers, total_trades, script_path.name))
        print(f"  ✓ {script_path.name:30} ({n_tickers:2} tickers, {total_trades:6} trades)")

    print("")
    print("=== Summary ===")
    print(f"Categories found: {len(generated)}")
    print(f"SLURM scripts:    {len(generated)}")
    print(f"Output dir:       {jobs_dir}")
    print("")
    print("Next steps:")
    print("  1. bash hpc/deploy.sh                  # deploy to HPC")
    print("  2. ssh mtk9va@login.hpc.virginia.edu")
    print("  3. cd /scratch/mtk9va/kalshi_v5")
    print("  4. sbatch hpc/train_mm_all.slurm       # submit all jobs")
    print("  5. squeue -u mtk9va                    # monitor jobs")


if __name__ == "__main__":
    main()
