"""Train MM PPO agent on historical Kalshi trade data with orderbook integration.

Uses the market-making environment (MMEnv) with:
- Subpenny pricing for queue priority (+0.001 bid, -0.001 ask)
- Market metadata loader for tick size validation
- 20-dimensional observation space with orderbook and realized-vol features
- PPO algorithm from Stable Baselines3

Usage:
    python -m rl_bot.mm_train --data output/rl_kalshi_trades_3mo.parquet
    python -m rl_bot.mm_train --category KXBTC --timesteps 500000
    python -m rl_bot.mm_train --category all --ppo-epochs 10
"""
import argparse
import logging
import sys
from pathlib import Path

import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_metadata import MarketMetadataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mm_train")

DEFAULT_DATA = "output/rl_kalshi_trades_3mo.parquet"
DEFAULT_MARKETS = "output/rl_all_markets_3mo.parquet"
DEFAULT_ORDERBOOKS = "output/mm_orderbooks.parquet"


def main():
    parser = argparse.ArgumentParser(description="Train MM PPO agent")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path to trades parquet file",
    )
    parser.add_argument(
        "--markets",
        type=str,
        default=DEFAULT_MARKETS,
        help="Path to markets metadata parquet",
    )
    parser.add_argument(
        "--orderbooks",
        type=str,
        default=None,
        help="Path to orderbooks parquet (optional)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Market category prefix (e.g., KXBTC, FEDHIKE) or 'all'",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Training timesteps per category",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=10,
        help="PPO epochs per update",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="mm_ppo",
        help="Run name for checkpoints/logs",
    )
    parser.add_argument(
        "--no-subpenny",
        action="store_true",
        help="Disable subpenny pricing",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="Temporal split date (e.g. 2026-04-23). Train on data before, test on data after.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which side of the split to use (default: train)",
    )
    args = parser.parse_args()

    # Load data — use scan + filter + collect for predicate pushdown when
    # a single category is requested. Avoids reading the full 24M-row parquet
    # into memory for each per-category SLURM job.
    log.info(f"Loading trades from {args.data}")
    if args.category != "all":
        # Lazy scan with predicate pushdown — only reads matching row groups
        trades = (
            pl.scan_parquet(args.data)
            .filter(pl.col("ticker").str.starts_with(args.category))
            .collect()
        )
        log.info(f"Loaded {len(trades):,} trades for category {args.category}")
        if len(trades) == 0:
            log.error(f"No trades found for category {args.category}")
            sys.exit(1)
    else:
        trades = pl.read_parquet(args.data)
        log.info(f"Loaded {len(trades):,} trades")

    # Discover unique tickers
    tickers = trades["ticker"].unique().sort().to_list()
    log.info(f"Found {len(tickers):,} unique tickers")

    # Load market metadata
    log.info("Loading market metadata")
    metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=args.markets)
    metadata = metadata_loader.load_metadata(tickers)
    log.info(f"Loaded metadata for {len(metadata):,} markets")

    # Load orderbook data if provided
    orderbooks_df = None
    if args.orderbooks and Path(args.orderbooks).exists():
        log.info(f"Loading orderbooks from {args.orderbooks}")
        if args.category != "all":
            orderbooks_df = (
                pl.scan_parquet(args.orderbooks)
                .filter(pl.col("ticker").str.starts_with(args.category))
                .collect()
            )
        else:
            orderbooks_df = pl.read_parquet(args.orderbooks)
        log.info(f"Loaded {len(orderbooks_df):,} orderbook snapshots")
    elif args.orderbooks:
        log.warning(f"Orderbooks file not found: {args.orderbooks}")

    # Preprocess trades into windows
    log.info("Preprocessing trades into time windows")
    if args.split_date:
        log.info(f"Temporal split: {args.split_mode} (cutoff={args.split_date})")
    windows = preprocess_mm_data(
        trades, orderbooks_df=orderbooks_df,
        split_date=args.split_date, split_mode=args.split_mode,
    )
    log.info(f"Created {len(windows):,} time windows")

    # Create environment
    config = MMConfig(
        max_inventory=20,
        quote_size=1,
        subpenny_enabled=not args.no_subpenny,
        api_environment="demo",  # Not used for training
    )

    def make_env():
        return MMEnv(ticker_data=windows, cfg=config, metadata_loader=metadata_loader)

    env = DummyVecEnv([make_env])
    log.info("Created MM environment with 20-dim observation space")
    log.info(f"Subpenny pricing: {'enabled' if config.subpenny_enabled else 'disabled'}")

    # Create PPO agent
    log.info("Initializing PPO agent")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=args.ppo_epochs,
        gamma=0.90,  # aggressive immediate-fill discount
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=None,  # disabled; install tensorboard to re-enable
    )

    # Setup checkpointing
    checkpoint_dir = Path("rl_bot/mm_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.run_name}_{args.category}",
    )

    # Train
    log.info(f"Starting training for {args.timesteps:,} timesteps")
    log.info(f"PPO epochs per update: {args.ppo_epochs}")
    log.info(f"Checkpoints: {checkpoint_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = checkpoint_dir / f"{args.run_name}_{args.category}_final.zip"
    model.save(final_path)
    log.info(f"Training complete. Final model: {final_path}")

    # Print summary stats
    log.info("\n=== Training Summary ===")
    log.info(f"Category: {args.category}")
    log.info(f"Timesteps: {args.timesteps:,}")
    log.info(f"Windows: {len(windows):,}")
    log.info(f"Tickers: {len(tickers):,}")
    log.info(f"Subpenny: {config.subpenny_enabled}")
    log.info(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
