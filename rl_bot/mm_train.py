"""Train MM PPO agent on historical Kalshi trade data with orderbook integration.

Uses the market-making environment (MMEnv) with:
- Subpenny pricing for queue priority (+0.001 bid, -0.001 ask)
- Market metadata loader for tick size validation
- 16-dimensional observation space with orderbook features
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
    args = parser.parse_args()

    # Load data
    log.info(f"Loading trades from {args.data}")
    trades = pl.read_parquet(args.data)
    log.info(f"Loaded {len(trades):,} trades")

    # Filter by category if specified
    if args.category != "all":
        trades = trades.filter(pl.col("ticker").str.starts_with(args.category))
        log.info(f"Filtered to {len(trades):,} trades for category {args.category}")
        if len(trades) == 0:
            log.error(f"No trades found for category {args.category}")
            sys.exit(1)

    # Discover unique tickers
    tickers = trades["ticker"].unique().sort().to_list()
    log.info(f"Found {len(tickers):,} unique tickers")

    # Load market metadata
    log.info("Loading market metadata")
    metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=args.markets)
    metadata = metadata_loader.load_metadata(tickers)
    log.info(f"Loaded metadata for {len(metadata):,} markets")

    # Preprocess trades into windows
    log.info("Preprocessing trades into time windows")
    windows = preprocess_mm_data(trades)
    log.info(f"Created {len(windows):,} time windows")

    # Create environment
    config = MMConfig(
        max_inventory=20,
        quote_size=1,
        subpenny_enabled=not args.no_subpenny,
        api_environment="demo",  # Not used for training
    )

    def make_env():
        return MMEnv(windows=windows, config=config, metadata_loader=metadata_loader)

    env = DummyVecEnv([make_env])
    log.info("Created MM environment with 16-dim observation space")
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
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"rl_bot/mm_logs/{args.run_name}",
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
