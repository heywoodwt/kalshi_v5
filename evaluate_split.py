#!/usr/bin/env python3
"""Evaluate MM checkpoints on train and test splits to detect overfitting."""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_metadata import MarketMetadataLoader


def evaluate_on_split(
    model: PPO,
    ticker_data: dict[str, list[dict]],
    config: MMConfig,
    metadata_loader: MarketMetadataLoader,
    n_episodes: int = 3,
) -> dict:
    """Run deterministic rollouts and return aggregate metrics."""
    if not ticker_data or sum(len(v) for v in ticker_data.values()) == 0:
        return {"mean_reward": 0.0, "total_pnl": 0.0, "mean_pnl": 0.0, "n_episodes": 0, "win_rate": 0.0}

    env = MMEnv(ticker_data=ticker_data, cfg=config, metadata_loader=metadata_loader)

    episode_rewards = []
    episode_pnls = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_pnls.append(info.get("pnl", 0.0))

    n = len(episode_pnls)
    return {
        "mean_reward": float(np.mean(episode_rewards)) if n else 0.0,
        "total_pnl": float(np.sum(episode_pnls)) if n else 0.0,
        "mean_pnl": float(np.mean(episode_pnls)) if n else 0.0,
        "n_episodes": n,
        "win_rate": float(np.mean([p > 0 for p in episode_pnls])) if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate train/test split performance")
    parser.add_argument("--checkpoints", type=str, default="rl_bot/mm_checkpoints")
    parser.add_argument("--data", type=str, default="output/rl_kalshi_trades_3mo.parquet")
    parser.add_argument("--markets", type=str, default="output/rl_all_markets_3mo.parquet")
    parser.add_argument("--split-date", type=str, default="2026-04-23")
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--output", type=str, default="hpc/mm_results/split_eval.csv")
    args = parser.parse_args()

    # Load data once
    print(f"Loading trades from {args.data}...")
    trades = pl.read_parquet(args.data)
    print(f"Loaded {len(trades):,} trades")

    metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=args.markets)

    config = MMConfig(
        max_inventory=20,
        quote_size=1,
        subpenny_enabled=True,
        api_environment="demo",
    )

    # Find all final checkpoints
    ckpt_dir = Path(args.checkpoints)
    checkpoints = sorted(ckpt_dir.glob("*_final.zip"))
    print(f"Found {len(checkpoints)} checkpoints\n")

    results = []

    for i, ckpt_path in enumerate(checkpoints):
        # Extract category from filename (e.g. mm_KXBTC_KXBTC_final.zip -> KXBTC)
        stem = ckpt_path.stem.replace("_final", "")
        # Handle both "mm_CAT_CAT" and "CAT" patterns
        parts = stem.split("_")
        if parts[0] == "mm" and len(parts) >= 3:
            category = parts[1]
        else:
            category = parts[0]

        print(f"[{i+1}/{len(checkpoints)}] {category}...", end=" ", flush=True)

        # Filter trades for this category
        cat_trades = trades.filter(pl.col("ticker").str.starts_with(category))
        if len(cat_trades) == 0:
            print("SKIP (no trades)")
            continue

        tickers = cat_trades["ticker"].unique().sort().to_list()
        metadata_loader.load_metadata(tickers)

        # Preprocess train and test splits
        train_data = preprocess_mm_data(cat_trades, split_date=args.split_date, split_mode="train")
        test_data = preprocess_mm_data(cat_trades, split_date=args.split_date, split_mode="test")

        train_windows = sum(len(v) for v in train_data.values())
        test_windows = sum(len(v) for v in test_data.values())

        # Load model
        try:
            model = PPO.load(ckpt_path)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            continue

        # Evaluate on both splits
        train_metrics = evaluate_on_split(model, train_data, config, metadata_loader, args.n_episodes)
        test_metrics = evaluate_on_split(model, test_data, config, metadata_loader, args.n_episodes)

        row = {
            "category": category,
            "train_windows": train_windows,
            "test_windows": test_windows,
            "train_mean_reward": train_metrics["mean_reward"],
            "train_mean_pnl": train_metrics["mean_pnl"],
            "train_win_rate": train_metrics["win_rate"],
            "test_mean_reward": test_metrics["mean_reward"],
            "test_mean_pnl": test_metrics["mean_pnl"],
            "test_win_rate": test_metrics["win_rate"],
            # Overfitting ratio: train/test reward. >1 = overfit, ~1 = good
            "overfit_ratio": (
                train_metrics["mean_reward"] / test_metrics["mean_reward"]
                if test_metrics["mean_reward"] != 0 else float("inf")
            ),
        }
        results.append(row)

        print(
            f"train=${train_metrics['mean_pnl']:+.2f} "
            f"test=${test_metrics['mean_pnl']:+.2f} "
            f"overfit={row['overfit_ratio']:.2f}x"
        )

    # Save results
    results_df = pl.DataFrame(results)
    results_df.write_csv(args.output)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Categories evaluated: {len(results_df)}")
    if len(results_df) > 0:
        print(f"Mean train PnL:  ${results_df['train_mean_pnl'].mean():+.4f}")
        print(f"Mean test PnL:   ${results_df['test_mean_pnl'].mean():+.4f}")
        print(f"Mean train win%: {results_df['train_win_rate'].mean():.1%}")
        print(f"Mean test win%:  {results_df['test_win_rate'].mean():.1%}")

        # Flag categories with severe overfitting
        overfit = results_df.filter(
            (pl.col("overfit_ratio") > 2.0) & (pl.col("overfit_ratio") != float("inf"))
        )
        if len(overfit) > 0:
            print(f"\n⚠ {len(overfit)} categories with overfit_ratio > 2x:")
            for row in overfit.sort("overfit_ratio", descending=True).head(10).iter_rows(named=True):
                print(f"  {row['category']:30s} {row['overfit_ratio']:.1f}x")


if __name__ == "__main__":
    main()
