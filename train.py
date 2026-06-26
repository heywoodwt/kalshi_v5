"""Train the RL agent on historical Kalshi trade data.

Runs multiple replay epochs over the same dataset with curriculum learning:
  - Epoch 1-2: high exploration (eps 1.0 -> 0.10), agent learns market dynamics
  - Epoch 3+:  low exploration (eps 0.10 -> 0.05), agent refines strategy

Each epoch resumes from the previous checkpoint so the replay buffer,
Q-network weights, and optimizer state carry over.

Usage:
    python train.py                                          # defaults
    python train.py --data output/trades.parquet --epochs 10
    python train.py --resume rl_bot/checkpoints/epoch_3.pt   # continue training
"""
import argparse
import csv
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from rl_bot.agent import DQNAgent
from rl_bot.btc_data import BTCDataPoller
from rl_bot.config import ACTION_HOLD, RLConfig, decode_action
from rl_bot.environment import TradingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rl_train")

_DEFAULT_DATA = "output/rl_kalshi_trades_3mo.parquet"

# Regex to extract expiry hour from KXBTCD tickers
_EXPIRY_HOUR_RE = re.compile(r"KXBTCD-\d{2}[A-Z]{3}(\d{2})")
# Regex to extract strike price
_STRIKE_RE = re.compile(r"-[TB](\d+\.?\d*)")


def _extract_strike(ticker: str) -> float | None:
    m = _STRIKE_RE.search(ticker)
    return float(m.group(1)) if m else None


def _parse_expiry_hour(ticker: str) -> int:
    m = _EXPIRY_HOUR_RE.search(ticker)
    if m:
        return int(m.group(1))
    try:
        return int(ticker[-2:])
    except (ValueError, IndexError):
        return 20


def _estimate_tte(ticker: str, current_time: datetime) -> float:
    target_hour = _parse_expiry_hour(ticker)
    current_hour = current_time.hour + current_time.minute / 60.0
    hours_diff = target_hour - current_hour
    if hours_diff < 0:
        hours_diff += 24
    return max(0.1, hours_diff)


def _preprocess_raw_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Convert raw Kalshi trades into 1-min bars."""
    # Filter to daily BTC markets
    df = df.filter(pl.col("ticker").str.starts_with("KXBTCD"))
    log.info("  Filtered to KXBTCD: %d trades", df.height)

    # Parse timestamps and truncate to 1-min
    df = df.with_columns(
        pl.col("created_time")
        .str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ", time_zone="UTC")
        .dt.truncate("1m")
        .alias("timestamp")
    )

    # Aggregate to 1-min bars per ticker (VWAP + volume)
    bars = df.group_by(["timestamp", "ticker"]).agg([
        (pl.col("yes_price") * pl.col("count")).sum().alias("_vwap_num"),
        pl.col("count").sum().alias("volume"),
    ]).with_columns(
        (pl.col("_vwap_num") / pl.col("volume")).alias("yes_price"),
    ).drop("_vwap_num")

    # Estimate BTC spot per minute from strike prices
    strikes = df.with_columns(
        pl.col("ticker").map_elements(
            lambda t: _extract_strike(t), return_dtype=pl.Float64
        ).alias("strike")
    ).filter(
        pl.col("strike").is_not_null()
    ).group_by("timestamp").agg([
        (pl.col("strike") * pl.col("count")).sum().alias("_sw"),
        pl.col("count").sum().alias("_w"),
    ]).with_columns(
        (pl.col("_sw") / pl.col("_w")).alias("spot_price"),
    ).select(["timestamp", "spot_price"])

    bars = bars.join(strikes, on="timestamp", how="left")
    bars = bars.sort("timestamp").with_columns(
        pl.col("spot_price").forward_fill().backward_fill(),
    )
    bars = bars.with_columns([
        pl.lit(0.0).alias("funding_rate"),
        pl.lit(0).cast(pl.Int64).alias("open_interest"),
    ])
    bars = bars.sort(["timestamp", "ticker"])
    return bars


def _run_epoch(
    df: pl.DataFrame,
    timestamps: list,
    cfg: RLConfig,
    agent: DQNAgent,
    env: TradingEnv,
    btc_poller: BTCDataPoller,
    epoch: int,
    csv_writer,
) -> dict:
    """Run one training epoch over the dataset. Returns stats dict."""
    total_trades = 0
    total_rewards = 0.0
    step_count = 0
    prev_states: dict[str, tuple] = {}

    for ts_idx, ts in enumerate(timestamps):
        batch = df.filter(pl.col("timestamp") == ts)

        # Feed BTC spot data
        spot_price = batch["spot_price"][0]
        funding_rate = batch["funding_rate"][0]
        btc_poller._on_spot_update(spot_price)
        btc_poller._on_funding_update(funding_rate)

        # Feed each ticker's data into the environment
        for row in batch.iter_rows(named=True):
            ticker = row["ticker"]
            yes_price = row["yes_price"] / 100.0 if row["yes_price"] > 1.0 else row["yes_price"]
            volume = int(row["volume"])
            tte = _estimate_tte(ticker, ts)
            env.on_ticker(ticker, yes_price, tte)
            if volume > 0:
                env.on_trade(ticker, yes_price, volume)

        # Agent decisions
        active_markets = env.get_active_markets()
        if not active_markets:
            agent.step_count += 1
            step_count += 1
            continue

        circuit_active = env.is_circuit_breaker_active()

        for ticker in active_markets:
            state = env.get_state(ticker)
            mask = env.get_mask(ticker)

            # Circuit breaker: mask all buy actions
            if circuit_active:
                for i in range(18):
                    mask[i] = 0.0

            # Store transition from previous step
            if ticker in prev_states:
                prev_state, prev_action = prev_states[ticker]
                agent.store_transition(prev_state, prev_action, 0.0, state, False)

            # Select and execute action
            action = agent.select_action(state, mask)
            raw_pnl_before = env.pnl_tracker.daily_pnl()
            next_state, reward, done = env.step(ticker, action)
            raw_pnl_after = env.pnl_tracker.daily_pnl()
            raw_pnl_delta = raw_pnl_after - raw_pnl_before

            if abs(raw_pnl_delta) > 1e-9:
                # A position was closed — store normalized reward for training
                agent.store_transition(state, action, reward, next_state, done)
                prev_states.pop(ticker, None)
                total_rewards += raw_pnl_delta
                total_trades += 1
            else:
                prev_states[ticker] = (state, action)

            # CSV logging
            decoded = decode_action(action)
            if isinstance(decoded, tuple):
                action_str = f"BUY_{decoded[0].upper()}_{decoded[1]}_AT_{int(decoded[2]*100)}c"
            else:
                action_str = decoded.upper()

            csv_writer.writerow([
                epoch,
                ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                ticker, action_str, f"{agent.epsilon():.4f}",
                env.pnl_tracker.get_position(ticker),
                f"{raw_pnl_delta:.6f}", f"{raw_pnl_after:.6f}",
                agent.step_count, f"{env._markets[ticker].last_price:.4f}",
                f"{spot_price:.2f}",
            ])

        # Train
        agent.step_count += 1
        step_count += 1
        loss = agent.train_step()

        # Periodic logging
        if step_count % 200 == 0:
            eps = agent.epsilon()
            daily = env.pnl_tracker.daily_pnl()
            loss_str = f"{loss:.6f}" if loss is not None else "warmup"
            log.info(
                "Epoch %d | Step %d/%d | eps=%.3f | pnl=$%.2f | trades=%d | loss=%s",
                epoch, step_count, len(timestamps), eps, daily, total_trades, loss_str,
            )

        # Checkpoint every 2000 steps
        if step_count % 2000 == 0:
            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = str(ckpt_dir / f"train_e{epoch}_s{agent.step_count}.pt")
            agent.save_checkpoint(path)

    # Close remaining positions at last known price
    for ticker in list(env._markets.keys()):
        pos = env.pnl_tracker.get_position(ticker)
        if pos != 0:
            close_action = 19 if pos > 0 else 20
            raw_before = env.pnl_tracker.daily_pnl()
            env.step(ticker, close_action)
            raw_delta = env.pnl_tracker.daily_pnl() - raw_before
            total_rewards += raw_delta
            total_trades += 1

    daily_pnl = env.pnl_tracker.daily_pnl()
    return {
        "epoch": epoch,
        "steps": step_count,
        "trades": total_trades,
        "pnl": daily_pnl,
        "avg_pnl_per_trade": daily_pnl / total_trades if total_trades > 0 else 0.0,
        "final_epsilon": agent.epsilon(),
        "buffer_size": len(agent._buffer),
    }


def train(
    data_path: str,
    epochs: int = 5,
    resume_path: str | None = None,
    max_daily_loss: float = 999999.0,
) -> None:
    """Train the RL agent over multiple epochs of historical data.

    Args:
        data_path: path to parquet trade data
        epochs: number of passes over the dataset
        resume_path: checkpoint to resume from
        max_daily_loss: circuit breaker threshold (default: disabled)
    """
    # Load and preprocess data once
    log.info("Loading data from %s...", data_path)
    df = pl.read_parquet(data_path)
    if "created_time" in df.columns and "timestamp" not in df.columns:
        df = _preprocess_raw_trades(df)
    else:
        df = df.sort("timestamp")

    n_rows = df.height
    timestamps = df["timestamp"].unique().sort().to_list()
    n_tickers = df["ticker"].n_unique()
    log.info("Data: %d bars, %d tickers, %d timesteps", n_rows, n_tickers, len(timestamps))

    # Curriculum: first 2 epochs explore more, then refine
    #   Epoch 1: eps 1.00 -> 0.20 (broad exploration)
    #   Epoch 2: eps 0.20 -> 0.10 (narrowing)
    #   Epoch 3+: eps 0.10 -> 0.05 (exploitation)
    curriculum = [
        {"eps_start": 1.00, "eps_end": 0.20, "decay_frac": 0.8},  # epoch 1
        {"eps_start": 0.20, "eps_end": 0.10, "decay_frac": 0.6},  # epoch 2
    ]
    # Epochs 3+ all use the same refinement schedule
    refine = {"eps_start": 0.10, "eps_end": 0.05, "decay_frac": 0.5}

    # CSV log for all epochs
    csv_path = Path("output/rl_training_log.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "timestamp", "ticker", "action", "epsilon", "position",
        "reward", "cumulative_pnl", "step", "market_price", "btc_spot",
    ])

    # Track results per epoch
    all_results = []

    for epoch in range(1, epochs + 1):
        # Pick curriculum schedule
        if epoch <= len(curriculum):
            sched = curriculum[epoch - 1]
        else:
            sched = refine

        decay_steps = int(len(timestamps) * sched["decay_frac"])

        cfg = RLConfig(
            eps_start=sched["eps_start"],
            eps_end=sched["eps_end"],
            eps_decay_steps=decay_steps,
            warmup_steps=min(200, len(timestamps) // 10),
            decision_interval_s=0.0,
            checkpoint_freq=2000,
            max_daily_loss=max_daily_loss,
            replay_capacity=500_000,  # larger buffer for multi-epoch
        )

        btc_poller = BTCDataPoller(poll_interval_s=999999)
        env = TradingEnv(cfg, btc_poller)
        agent = DQNAgent(cfg)

        # Resume from previous epoch's checkpoint or user-provided path
        ckpt_to_load = resume_path if (epoch == 1 and resume_path) else None
        if epoch > 1:
            # Load the previous epoch's final checkpoint
            prev_ckpt = Path(cfg.checkpoint_dir) / f"train_epoch_{epoch - 1}_final.pt"
            if prev_ckpt.exists():
                ckpt_to_load = str(prev_ckpt)

        if ckpt_to_load and Path(ckpt_to_load).exists():
            agent.load_checkpoint(ckpt_to_load)
            # Keep loaded step_count so epsilon schedule resumes correctly
            # (do not reset to 0 which would restart exploration)
            log.info("Epoch %d: loaded checkpoint %s", epoch, ckpt_to_load)

        log.info(
            "=== Epoch %d/%d | eps %.2f -> %.2f over %d steps ===",
            epoch, epochs, sched["eps_start"], sched["eps_end"], decay_steps,
        )

        result = _run_epoch(
            df, timestamps, cfg, agent, env, btc_poller, epoch, csv_writer,
        )
        all_results.append(result)

        # Save epoch-end checkpoint
        ckpt_dir = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = str(ckpt_dir / f"train_epoch_{epoch}_final.pt")
        agent.save_checkpoint(final_path)

        log.info(
            "Epoch %d complete: PnL=$%.2f | trades=%d | avg=$%.4f | eps=%.3f | buffer=%d",
            result["epoch"], result["pnl"], result["trades"],
            result["avg_pnl_per_trade"], result["final_epsilon"], result["buffer_size"],
        )

    csv_file.close()

    # Print summary table
    log.info("=" * 70)
    log.info("TRAINING COMPLETE")
    log.info("=" * 70)
    log.info("%-6s %12s %8s %12s %8s %8s", "Epoch", "PnL", "Trades", "Avg/Trade", "Epsilon", "Buffer")
    log.info("-" * 70)
    for r in all_results:
        log.info(
            "%-6d %12.2f %8d %12.4f %8.3f %8d",
            r["epoch"], r["pnl"], r["trades"],
            r["avg_pnl_per_trade"], r["final_epsilon"], r["buffer_size"],
        )
    log.info("=" * 70)
    log.info("Training log: %s", csv_path)
    log.info("Final checkpoint: %s", ckpt_dir / f"train_epoch_{epochs}_final.pt")


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on historical Kalshi data")
    parser.add_argument(
        "--data", default=_DEFAULT_DATA,
        help="Path to parquet data file (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--max-loss", type=float, default=999999.0,
        help="Circuit breaker max daily loss (default: disabled)",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        resume_path=args.resume,
        max_daily_loss=args.max_loss,
    )


if __name__ == "__main__":
    main()
