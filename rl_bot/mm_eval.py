"""Evaluate a trained MM PPO agent on out-of-sample data.

Loads a saved model checkpoint and runs deterministic rollouts on test data,
reporting per-category and aggregate metrics.

Usage:
    python -m rl_bot.mm_eval --model rl_bot/mm_checkpoints/split_test_KXBTC_final.zip \
        --data output/rl_kalshi_trades_3mo.parquet --split-date 2026-04-23 --category KXBTC
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_metadata import MarketMetadataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mm_eval")

DEFAULT_DATA = "output/rl_kalshi_trades_3mo.parquet"
DEFAULT_MARKETS = "output/rl_all_markets_3mo.parquet"


def main():
    parser = argparse.ArgumentParser(description="Evaluate MM PPO agent on test data")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .zip")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to trades parquet")
    parser.add_argument("--markets", type=str, default=DEFAULT_MARKETS, help="Path to markets parquet")
    parser.add_argument("--category", type=str, default="all", help="Category prefix or 'all'")
    parser.add_argument("--split-date", type=str, default="2026-04-23", help="Temporal split date")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (optional)")
    parser.add_argument("--no-subpenny", action="store_true", help="Disable subpenny pricing")
    parser.add_argument("--orderbooks", type=str, default=None, help="Path to orderbooks parquet")
    args = parser.parse_args()

    # Load model
    log.info(f"Loading model from {args.model}")
    model = PPO.load(args.model)

    # Load and filter data
    log.info(f"Loading trades from {args.data}")
    trades = pl.read_parquet(args.data)
    if args.category != "all":
        trades = trades.filter(pl.col("ticker").str.starts_with(args.category))
        log.info(f"Filtered to {len(trades):,} trades for category {args.category}")
        if len(trades) == 0:
            log.error(f"No trades found for category {args.category}")
            sys.exit(1)

    tickers = trades["ticker"].unique().sort().to_list()
    log.info(f"Found {len(tickers):,} unique tickers")

    # Load metadata
    metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=args.markets)
    metadata = metadata_loader.load_metadata(tickers)

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

    # Preprocess with test split
    log.info(f"Preprocessing test data (split_date={args.split_date})")
    windows = preprocess_mm_data(
        trades, orderbooks_df=orderbooks_df,
        split_date=args.split_date, split_mode="test",
    )
    total_windows = sum(len(v) for v in windows.values())
    log.info(f"Test set: {len(windows)} tickers, {total_windows} windows")

    if total_windows == 0:
        log.error("No test data after split. Check --split-date.")
        sys.exit(1)

    # Create environment — disable domain randomization for eval
    # (eval should reflect actual market conditions, not injected noise)
    config = MMConfig(
        max_inventory=20,
        quote_size=1,
        subpenny_enabled=not args.no_subpenny,
        api_environment="demo",
        domain_rand_spread_prob=0.0,
        domain_rand_volume_prob=0.0,
    )
    env = MMEnv(ticker_data=windows, cfg=config, metadata_loader=metadata_loader)

    # Run deterministic rollouts — one episode per ticker
    results = []
    for ticker in sorted(windows.keys()):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            # Deterministic action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        results.append({
            "ticker": ticker,
            "steps": steps,
            "total_reward": episode_reward,
            "realized_pnl": info.get("realized_pnl", 0.0),
            "unrealized_pnl": info.get("unrealized_pnl", 0.0),
            "total_pnl": info.get("pnl", 0.0),
            "final_inventory": info.get("inventory", 0),
            "fills_buy": info.get("fills_buy", 0),
            "fills_sell": info.get("fills_sell", 0),
        })

    # Build results dataframe
    results_df = pl.DataFrame(results)

    # Print per-ticker results
    log.info("\n=== Per-Ticker Results ===")
    for row in results_df.iter_rows(named=True):
        log.info(
            f"  {row['ticker']:30s}  reward={row['total_reward']:+8.4f}  "
            f"pnl={row['total_pnl']:+8.4f}  steps={row['steps']:5d}  "
            f"inv={row['final_inventory']:+3d}"
        )

    # Aggregate metrics
    n_episodes = len(results_df)
    mean_reward = results_df["total_reward"].mean()
    total_pnl = results_df["total_pnl"].sum()
    win_rate = (results_df["total_pnl"] > 0).sum() / max(n_episodes, 1)

    log.info("\n=== Aggregate Metrics (Out-of-Sample) ===")
    log.info(f"  Episodes:    {n_episodes}")
    log.info(f"  Mean reward: {mean_reward:+.4f}")
    log.info(f"  Total PnL:   {total_pnl:+.4f}")
    log.info(f"  Win rate:    {win_rate:.1%}")

    # Save CSV if requested
    output_path = args.output or f"output/mm_eval_{args.category}.csv"
    results_df.write_csv(output_path)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
