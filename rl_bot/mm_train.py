"""Market-Making PPO Training & Evaluation Script.

Trains a market-agnostic PPO agent on historical Kalshi trade data. Supports
training per category (e.g., KXBTC15M) or across all categories. Includes
evaluation mode to replay trained models and generate per-ticker PnL summaries.

Usage:
    # Train on single category
    python -m rl_bot.mm_train --data trades.parquet --category KXBTC15M

    # Train on all categories
    python -m rl_bot.mm_train --data trades.parquet --category all

    # Evaluate saved model
    python -m rl_bot.mm_train --data trades.parquet --category KXBTC15M --eval --checkpoint rl_bot/mm_checkpoints/KXBTC15M.zip

Key design:
  - O(n) complexity for data loading and preprocessing
  - Polars for all data operations
  - CSV logging per training step
  - Simple code suitable for future Rust translation
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_trades_for_mm


def list_categories(df: pl.DataFrame) -> list[str]:
    """Extract unique ticker prefixes using regex.

    Uses pattern ^([A-Z]+) to extract letter prefix from each ticker.
    For example: KXBTC15M-26JUN24-T17 -> KXBTC15M

    Args:
        df: Polars DataFrame with "ticker" column

    Returns:
        Sorted list of unique category prefixes

    Time complexity: O(n) where n = number of unique tickers
    """
    # Extract unique tickers
    unique_tickers = df["ticker"].unique().to_list()

    # Use regex to extract prefix (all uppercase letters before any dash/digit)
    categories = set()
    pattern = re.compile(r"^([A-Z]+)")

    for ticker in unique_tickers:
        match = pattern.match(ticker)
        if match:
            categories.add(match.group(1))

    return sorted(list(categories))


class MMCSVLogger(BaseCallback):
    """Custom SB3 callback that logs MM-specific state to CSV.

    Logs per step: step, timestamp, ticker, bid, ask, inventory, fills_buy,
    fills_sell, pnl, reward, half_spread, skew.

    Accesses the unwrapped MMEnv to extract state information.
    """

    def __init__(self, log_path: str):
        """Initialize CSV logger.

        Args:
            log_path: Path to output CSV file
        """
        super().__init__()
        self.log_path = log_path
        self.csv_file = None
        self.csv_writer = None

    def _on_training_start(self) -> None:
        """Open CSV file and write header."""
        # Create parent directory if needed
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        # Open CSV file
        self.csv_file = open(self.log_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write header
        self.csv_writer.writerow([
            "step",
            "timestamp",
            "ticker",
            "bid",
            "ask",
            "inventory",
            "fills_buy",
            "fills_sell",
            "pnl",
            "reward",
            "half_spread",
            "skew",
        ])
        self.csv_file.flush()

    def _on_step(self) -> bool:
        """Log current step state to CSV.

        Returns:
            True to continue training
        """
        # Access the unwrapped environment
        # training_env is a VecEnv wrapper, get first env and unwrap
        env = self.training_env.envs[0].unwrapped

        # Get info from the environment's last step
        # The info dict is populated by MMEnv.step()
        if hasattr(env, "_current_ticker"):
            # Log the step using environment's current state
            self.csv_writer.writerow([
                self.num_timesteps,
                env._current_timestamp,
                env._current_ticker,
                f"{env._current_bid:.4f}",
                f"{env._current_ask:.4f}",
                env._inventory,
                env._fills_buy,
                env._fills_sell,
                f"{env._realized_pnl + env._unrealized_pnl():.4f}",
                f"{0.0:.6f}",  # Reward not directly accessible
                f"{env._current_half_spread:.4f}",
                f"{env._current_skew:.4f}",
            ])

            # Flush every 100 steps to avoid buffering delays
            if self.num_timesteps % 100 == 0:
                self.csv_file.flush()

        return True

    def _on_training_end(self) -> None:
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()


def train_category(
    df: pl.DataFrame,
    category: str,
    cfg: MMConfig,
    run_name: str = "default",
) -> str:
    """Train PPO agent on a single market category.

    Pipeline:
      1. Filter to tickers starting with category prefix
      2. Find tickers with >= min_trades_per_ticker trades
      3. Preprocess into per-ticker, per-minute windows
      4. Create MMEnv with multi-ticker data
      5. Train PPO with config hyperparameters
      6. Save model to mm_checkpoints/{category}.zip

    Args:
        df: Raw trades DataFrame
        category: Ticker prefix (e.g., "KXBTC15M")
        cfg: MMConfig with training hyperparameters
        run_name: Experiment name for logging

    Returns:
        Path to saved model checkpoint

    Time complexity: O(n) where n = number of trades in category
    """
    print(f"\n[{category}] Starting training...")

    # Filter to category prefix
    cat_df = df.filter(pl.col("ticker").str.starts_with(category))
    print(f"[{category}] Found {len(cat_df)} trades across all tickers")

    # Find tickers with enough liquidity
    ticker_counts = (
        cat_df.group_by("ticker")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= cfg.min_trades_per_ticker)
    )
    tickers = ticker_counts["ticker"].to_list()
    print(f"[{category}] {len(tickers)} tickers with >= {cfg.min_trades_per_ticker} trades")

    if len(tickers) == 0:
        print(f"[{category}] No tickers meet liquidity threshold, skipping")
        return ""

    # Filter to liquid tickers
    filtered_df = cat_df.filter(pl.col("ticker").is_in(tickers))

    # Preprocess into per-ticker windows
    print(f"[{category}] Preprocessing trades into 1-minute windows...")
    ticker_data = preprocess_trades_for_mm(filtered_df)
    total_windows = sum(len(windows) for windows in ticker_data.values())
    print(f"[{category}] Generated {total_windows} total windows across {len(ticker_data)} tickers")

    # Create environment
    print(f"[{category}] Creating MMEnv...")
    env = MMEnv(ticker_data, cfg)

    # Create PPO model
    print(f"[{category}] Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs_ppo,
        verbose=1,
    )

    # Setup CSV logging
    log_dir = Path("rl_bot/mm_logs") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{category}.csv"
    csv_callback = MMCSVLogger(str(log_path))

    # Train
    print(f"[{category}] Training for {cfg.total_timesteps} steps...")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=csv_callback,
        progress_bar=False,  # Disabled to avoid tqdm dependency
    )

    # Save model
    checkpoint_dir = Path("rl_bot/mm_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{category}.zip"
    model.save(str(checkpoint_path))
    print(f"[{category}] Model saved to {checkpoint_path}")

    return str(checkpoint_path)


def eval_model(
    checkpoint_path: str,
    df: pl.DataFrame,
    category: str,
    cfg: MMConfig,
) -> None:
    """Evaluate trained model on historical data.

    Loads saved model, creates env, runs deterministic episodes, and prints
    per-ticker PnL summary.

    Args:
        checkpoint_path: Path to saved model (.zip)
        df: Raw trades DataFrame
        category: Ticker prefix
        cfg: MMConfig (should match training config)

    Time complexity: O(n) where n = number of trades in category
    """
    print(f"\n[{category}] Evaluating model from {checkpoint_path}...")

    # Filter and preprocess data (same as training)
    cat_df = df.filter(pl.col("ticker").str.starts_with(category))
    ticker_counts = (
        cat_df.group_by("ticker")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= cfg.min_trades_per_ticker)
    )
    tickers = ticker_counts["ticker"].to_list()

    if len(tickers) == 0:
        print(f"[{category}] No tickers meet liquidity threshold")
        return

    filtered_df = cat_df.filter(pl.col("ticker").is_in(tickers))
    ticker_data = preprocess_trades_for_mm(filtered_df)

    # Create environment
    env = MMEnv(ticker_data, cfg)

    # Load model
    print(f"[{category}] Loading model...")
    model = PPO.load(checkpoint_path, env=env)

    # Run deterministic episodes
    print(f"[{category}] Running evaluation...")
    ticker_pnls: dict[str, float] = {}
    total_episodes = 0
    total_steps = 0

    # Run until we've seen all tickers
    obs, _ = env.reset()
    done = False

    while total_episodes < len(tickers) * 2:  # Run 2 episodes per ticker
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_steps += 1

        if done or truncated:
            # Record PnL for this ticker
            ticker = info.get("ticker", "unknown")
            pnl = info.get("pnl", 0.0)

            if ticker not in ticker_pnls:
                ticker_pnls[ticker] = 0.0
            ticker_pnls[ticker] += pnl

            total_episodes += 1
            obs, _ = env.reset()

    # Print summary
    print(f"\n[{category}] Evaluation Results:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"\nPer-ticker PnL:")
    total_pnl = 0.0
    for ticker in sorted(ticker_pnls.keys()):
        pnl = ticker_pnls[ticker]
        total_pnl += pnl
        print(f"  {ticker}: ${pnl:.2f}")
    print(f"\nTotal PnL: ${total_pnl:.2f}")
    print(f"Average PnL per ticker: ${total_pnl / max(len(ticker_pnls), 1):.2f}")


def main() -> None:
    """Main entry point with argparse CLI."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate MM PPO agent on Kalshi trade data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input parquet file with trade data",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Market category prefix (e.g., KXBTC15M) or 'all' for all categories",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (overrides config default)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable evaluation mode (requires --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to saved model for evaluation",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.eval and not args.checkpoint:
        parser.error("--eval requires --checkpoint")

    # Load data
    print(f"Loading data from {args.data}...")
    df = pl.read_parquet(args.data)
    print(f"Loaded {len(df)} trades")

    # Create config with custom timesteps
    cfg = MMConfig(total_timesteps=args.timesteps)

    # Evaluation mode
    if args.eval:
        eval_model(args.checkpoint, df, args.category, cfg)
        return

    # Training mode
    if args.category == "all":
        # Train on all categories
        categories = list_categories(df)
        print(f"\nDiscovered {len(categories)} categories: {categories}")

        for category in categories:
            train_category(df, category, cfg, args.run_name)
    else:
        # Train single category
        train_category(df, args.category, cfg, args.run_name)


if __name__ == "__main__":
    main()
