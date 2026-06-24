"""Replay the RL agent on saved historical data (paper mode).

Reads a parquet file of Kalshi BTC market data and feeds it through
the TradingEnv + DQNAgent as if it were a live WebSocket stream.
The agent learns online as it replays — same as live, just faster.

Supports multi-epoch training: the agent replays the same data N times,
with epsilon decaying across all epochs. The agent's network, optimizer,
and replay buffer persist across epochs; only the environment resets.

Supports two data formats:
  1. Pre-aggregated 1-min bars (columns: timestamp, ticker, yes_price, volume,
     spot_price, funding_rate, open_interest)
  2. Raw Kalshi trades (columns: trade_id, ticker, count, yes_price, no_price,
     taker_side, created_time) — auto-detected and resampled to 1-min bars

Usage:
    python -m rl_bot.replay                                    # default data
    python -m rl_bot.replay --data path/to/data.parquet        # custom data
    python -m rl_bot.replay --speed 0                          # no delay (fastest)
    python -m rl_bot.replay --epochs 10                        # 10 passes over data
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
from rl_bot.exploration import (
    ExplorationStrategy,
    FastLinearDecay,
    ExponentialDecay,
    LogarithmicDecay,
    EpisodeBased,
    ActionLocal,
    ParameterNoise,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rl_replay")

# Default path to the saved training data
_DEFAULT_DATA = "output/historical_trades_apr18_22.parquet"

# Number of rows between agent decisions (data is 1-min bars,
# decision cadence is 30s, so decide every row for denser learning)
_DECISION_EVERY = 1

# Regex to extract strike price from ticker (e.g. KXBTCD-26APR1803-T77099.99 -> 77099.99)
_STRIKE_RE = re.compile(r"-[TB](\d+\.?\d*)")

# Regex to extract expiry hour from KXBTCD tickers
# KXBTCD-26APR1803-T77099.99 -> the "03" after the day+month = hour 03
# Format: KXBTCD-<day><MON><hour><subhour>-<strike>
_EXPIRY_HOUR_RE = re.compile(r"KXBTCD-\d{2}[A-Z]{3}(\d{2})")


def create_exploration_strategy(
    strategy_name: str,
    n_rows: int,
) -> ExplorationStrategy:
    """Factory to instantiate strategy by name.

    Args:
        strategy_name: One of the registered strategy names
        n_rows: Total dataset size (for decay_steps tuning)

    Returns:
        Configured ExplorationStrategy instance

    Raises:
        ValueError: If strategy_name is unknown
    """
    strategies = {
        "fast_linear": FastLinearDecay({
            "eps_start": 0.5,
            "eps_end": 0.05,
            "decay_steps": n_rows // 10,
        }),
        "exponential": ExponentialDecay({
            "eps_start": 0.8,
            "eps_end": 0.05,
            "decay_rate": 0.9990,
        }),
        "logarithmic": LogarithmicDecay({
            "eps_start": 0.8,
            "eps_end": 0.05,
            "decay_steps": n_rows // 2,
        }),
        "episode": EpisodeBased({
            "eps_start": 0.8,
            "eps_end": 0.05,
            "decay_rate": 0.99,
        }),
        "action_local": ActionLocal({
            "eps_start": 0.6,
            "eps_end": 0.05,
            "decay_steps": n_rows // 5,
        }),
        "parameter_noise": ParameterNoise({
            "noise_std_start": 0.1,
            "noise_std_end": 0.01,
            "decay_steps": n_rows // 5,
        }),
    }

    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Valid options: {list(strategies.keys())}"
        )

    return strategies[strategy_name]


def _extract_strike(ticker: str) -> float | None:
    """Parse BTC strike price from ticker name."""
    m = _STRIKE_RE.search(ticker)
    if m:
        return float(m.group(1))
    return None


def _parse_expiry_hour(ticker: str) -> int:
    """Extract the target hour from a ticker.

    Handles both formats:
      - Old: KXBTCD-26APR1817 -> 17 (last 2 digits)
      - New: KXBTCD-26APR1803-T77099.99 -> 03 (2 digits after day+month)
    """
    # Try new format first (with strike suffix)
    m = _EXPIRY_HOUR_RE.search(ticker)
    if m:
        return int(m.group(1))
    # Fallback: last 2 digits of ticker
    try:
        return int(ticker[-2:])
    except (ValueError, IndexError):
        return 20  # fallback


def _estimate_tte(ticker: str, current_time: datetime) -> float:
    """Estimate hours-to-expiry from ticker name and current timestamp."""
    target_hour = _parse_expiry_hour(ticker)
    current_hour = current_time.hour + current_time.minute / 60.0
    hours_diff = target_hour - current_hour
    if hours_diff < 0:
        hours_diff += 24
    return max(0.1, hours_diff)


def _preprocess_raw_trades(df: pl.DataFrame) -> pl.DataFrame:
    """Convert raw Kalshi trades into 1-min bars for replay.

    Input columns: trade_id, ticker, count, yes_price, no_price,
                   taker_side, created_time
    Output columns: timestamp, ticker, yes_price, volume, spot_price,
                    funding_rate, open_interest

    Steps:
      1. Filter to KXBTCD (daily BTC) markets only
      2. Parse created_time to datetime, truncate to 1-min
      3. Aggregate per (minute, ticker): VWAP yes_price, sum volume
      4. Estimate BTC spot from volume-weighted median of strike prices
    """
    log.info("Detected raw trades format — preprocessing to 1-min bars...")

    # 1. Filter to daily BTC markets (KXBTCD prefix)
    df = df.filter(pl.col("ticker").str.starts_with("KXBTCD"))
    log.info("  Filtered to KXBTCD: %d trades", df.height)

    # 2. Parse timestamps and truncate to 1-min buckets
    df = df.with_columns(
        pl.col("created_time")
        .str.to_datetime("%Y-%m-%dT%H:%M:%S%.fZ", time_zone="UTC")
        .dt.truncate("1m")
        .alias("timestamp")
    )

    # 3. Aggregate to 1-min bars per ticker
    #    - VWAP for yes_price (volume-weighted average)
    #    - sum of count as volume
    bars = df.group_by(["timestamp", "ticker"]).agg([
        # VWAP: sum(price * volume) / sum(volume)
        (pl.col("yes_price") * pl.col("count")).sum().alias("_vwap_num"),
        pl.col("count").sum().alias("volume"),
    ]).with_columns(
        (pl.col("_vwap_num") / pl.col("volume")).alias("yes_price"),
    ).drop("_vwap_num")

    # 4. Estimate BTC spot per minute from strike prices of traded contracts
    #    Extract strike from each ticker, compute volume-weighted median per minute
    strikes = df.with_columns(
        pl.col("ticker").map_elements(
            lambda t: _extract_strike(t), return_dtype=pl.Float64
        ).alias("strike")
    ).filter(
        pl.col("strike").is_not_null()
    ).group_by("timestamp").agg([
        # Volume-weighted mean of strikes as spot estimate
        (pl.col("strike") * pl.col("count")).sum().alias("_sw"),
        pl.col("count").sum().alias("_w"),
    ]).with_columns(
        (pl.col("_sw") / pl.col("_w")).alias("spot_price"),
    ).select(["timestamp", "spot_price"])

    # 5. Join spot estimate onto bars
    bars = bars.join(strikes, on="timestamp", how="left")

    # Fill any missing spot with forward-fill then backward-fill
    bars = bars.sort("timestamp").with_columns(
        pl.col("spot_price").forward_fill().backward_fill(),
    )

    # 6. Add placeholder columns the replay loop expects
    bars = bars.with_columns([
        pl.lit(0.0).alias("funding_rate"),
        pl.lit(0).cast(pl.Int64).alias("open_interest"),
    ])

    # 7. Final sort
    bars = bars.sort(["timestamp", "ticker"])

    n_tickers = bars["ticker"].n_unique()
    n_minutes = bars["timestamp"].n_unique()
    log.info(
        "  Resampled: %d bars, %d tickers, %d minutes",
        bars.height, n_tickers, n_minutes,
    )
    return bars


def _select_markets(
    active_markets: list[str],
    env: TradingEnv,
    cfg: RLConfig,
    prioritize_held: bool,
) -> list[str]:
    """Pick which markets the agent acts on this step.

    If max_markets_per_step is None, returns all active markets.
    If prioritize_held, guarantees markets with open positions are included,
    then fills remaining slots by volatility (highest first).
    Otherwise ranks all markets by volatility.
    """
    if cfg.max_markets_per_step is None:
        return active_markets
    if len(active_markets) <= cfg.max_markets_per_step:
        return active_markets

    if prioritize_held:
        # Guarantee held markets appear in selection
        held = [t for t in active_markets
                if env.pnl_tracker.get_position(t) != 0]
        held_set = set(held)
        remaining = [t for t in active_markets if t not in held_set]
        # Fill remaining slots by volatility
        vols = []
        for t in remaining:
            try:
                vol = env._markets[t].vol_tracker.get_metrics().get("vol", 0.0)
            except Exception:
                vol = 0.0
            vols.append((t, vol))
        vols.sort(key=lambda x: x[1], reverse=True)
        slots_left = max(0, cfg.max_markets_per_step - len(held))
        return held + [t for t, _ in vols[:slots_left]]

    # Default: rank by volatility only
    vols = []
    for t in active_markets:
        try:
            vol = env._markets[t].vol_tracker.get_metrics().get("vol", 0.0)
        except Exception:
            vol = 0.0
        vols.append((t, vol))
    vols.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in vols[:cfg.max_markets_per_step]]


def run_replay(
    data_path: str,
    speed: float = 0.0,
    checkpoint_path: str | None = None,
    max_daily_loss: float = 50.0,
    no_market_limit: bool = False,
    prioritize_held: bool = False,
    run_name: str = "",
    invert_policy: bool = False,
    no_fees: bool = False,
    n_epochs: int = 1,
    strategy_name: str | None = None,
) -> dict:
    """Replay historical data through the RL agent.

    Args:
        data_path: path to parquet file with 1-min bars or raw trades
        speed: seconds to sleep between steps (0 = no delay)
        checkpoint_path: optional path to load agent checkpoint
        max_daily_loss: circuit breaker threshold (default 50.0)
        no_market_limit: if True, act on ALL active markets (no cap)
        prioritize_held: if True, always include markets with open positions
        run_name: suffix for checkpoint dir and CSV log isolation
        invert_policy: if True, pick lowest Q-value (test signal direction)
        no_fees: if True, zero out all fees and spread
        n_epochs: number of passes over the data (default 1)
        strategy_name: exploration strategy name (default None = legacy linear)

    Returns:
        dict with summary stats: total_pnl, trades, steps, etc.
    """
    # ── Load data (once, reused across epochs) ──────────────────────
    df = pl.read_parquet(data_path)

    # Auto-detect raw trades vs pre-aggregated bars
    if "created_time" in df.columns and "timestamp" not in df.columns:
        df = _preprocess_raw_trades(df)
    else:
        df = df.sort("timestamp")

    n_rows = df.height
    tickers = df["ticker"].unique().sort().to_list()
    # n_steps = decision steps per epoch (unique timestamps, not bar count)
    n_steps = df["timestamp"].n_unique()
    # Total training steps across all epochs — drives epsilon schedule
    total_training_steps = n_steps * n_epochs
    time_range = (df["timestamp"].min(), df["timestamp"].max())
    log.info(
        "Loaded %d rows, %d tickers, %d steps/epoch, %d epochs (%d total steps)",
        n_rows, len(tickers), n_steps, n_epochs, total_training_steps,
    )
    log.info("  Time range: %s to %s", time_range[0], time_range[1])

    # ── Pre-group data by timestamp for O(1) lookup per step ────────
    # Without this, each epoch re-filters the full DataFrame (O(n) per step).
    # With pre-grouping, each step is O(1) dict lookup.
    timestamps = df["timestamp"].unique().sort().to_list()
    ts_groups: dict = {}
    for group in df.partition_by("timestamp", maintain_order=True):
        ts_groups[group["timestamp"][0]] = group

    # ── Config ──────────────────────────────────────────────────────
    # Epsilon decays over total_training_steps (all epochs), not one epoch.
    ckpt_dir = "rl_bot/checkpoints"
    csv_log = "output/rl_trades.csv"
    if run_name:
        ckpt_dir = f"rl_bot/checkpoints_{run_name}"
        csv_log = f"output/rl_trades_{run_name}.csv"

    cfg = RLConfig(
        eps_start=0.5,
        eps_end=0.05,
        # Explore for first 50% of total training steps across all epochs
        eps_decay_steps=total_training_steps // 2,
        warmup_steps=min(2000, n_steps // 5),
        decision_interval_s=0.0,  # not used in replay
        checkpoint_freq=500,
        max_daily_loss=max_daily_loss,
        # No extra trade penalty — fee-aware reward + execution model handle costs.
        trade_penalty=0.0,
        reward_scale=1.0,
        max_markets_per_step=None if no_market_limit else 5,
        checkpoint_dir=ckpt_dir,
        log_csv_path=csv_log,
        # Zero fees + zero spread to isolate directional signal
        maker_fee_rate=0.0 if no_fees else 0.0175,
        taker_fee_rate=0.0 if no_fees else 0.07,
        base_spread=0.0 if no_fees else 0.03,
    )
    # Enable observation normalization for replay (not live mode)
    cfg = cfg.__class__(**{**cfg.__dict__, "normalize_observations": True})

    # ── Create exploration strategy if provided ──
    strategy = None
    if strategy_name:
        strategy = create_exploration_strategy(strategy_name, total_training_steps)
        log.info("Using exploration strategy: %s", strategy_name)

    # ── Agent (persists across epochs: network, buffer, optimizer) ──
    agent = DQNAgent(cfg, exploration_strategy=strategy)
    if checkpoint_path and Path(checkpoint_path).exists():
        agent.load_checkpoint(checkpoint_path)
        log.info("Loaded checkpoint: %s (step %d)", checkpoint_path, agent.step_count)

    # ── CSV log (single file across all epochs, with epoch column) ──
    csv_suffix = f"_replay_{run_name}" if run_name else "_replay"
    csv_path = Path(cfg.log_csv_path).parent / f"rl{csv_suffix}_trades.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "timestamp", "ticker", "action", "epsilon", "position",
        "reward", "cumulative_pnl", "step", "market_price", "btc_spot",
    ])

    # ── Grand totals across all epochs ──────────────────────────────
    grand_trades = 0
    grand_pnl = 0.0
    step_count = 0  # local counter for logging/checkpointing

    # ── Epoch loop ──────────────────────────────────────────────────
    for epoch in range(n_epochs):
        log.info("=" * 60)
        log.info("EPOCH %d/%d  (agent step %d, eps=%.3f)",
                 epoch + 1, n_epochs, agent.step_count, agent.epsilon())
        log.info("=" * 60)

        # Fresh environment and BTC poller each epoch — positions reset,
        # market state resets, but agent (network + buffer) carries over
        btc_poller = BTCDataPoller(poll_interval_s=999999)
        env = TradingEnv(cfg, btc_poller)

        # Update episode count for EpisodeBased strategy
        if isinstance(strategy, EpisodeBased):
            strategy.episode_count = epoch

        epoch_trades = 0

        for ts_idx, ts in enumerate(timestamps):
            # O(1) lookup from pre-grouped dict
            batch = ts_groups[ts]

            # Feed BTC spot data from the first row (same across tickers)
            spot_price = batch["spot_price"][0]
            funding_rate = batch["funding_rate"][0]
            btc_poller._on_spot_update(spot_price)
            btc_poller._on_funding_update(funding_rate)

            # Feed each ticker's data into the environment
            for row in batch.iter_rows(named=True):
                ticker = row["ticker"]
                # Normalize yes_price from cents (0-100) to probability (0-1)
                yes_price = row["yes_price"] / 100.0 if row["yes_price"] > 1.0 else row["yes_price"]
                volume = int(row["volume"])
                tte = _estimate_tte(ticker, ts)
                env.on_ticker(ticker, yes_price, tte)
                if volume > 0:
                    env.on_trade(ticker, yes_price, volume)

            # Agent decision step (every _DECISION_EVERY rows)
            if ts_idx % _DECISION_EVERY != 0:
                continue

            active_markets = env.get_active_markets()
            if not active_markets:
                continue

            circuit_active = env.is_circuit_breaker_active()

            # Select which markets to act on this step
            markets_to_act = _select_markets(
                active_markets, env, cfg, prioritize_held,
            )

            for ticker in markets_to_act:
                state = env.get_state(ticker)
                mask = env.get_mask(ticker)

                # Circuit breaker: mask all buy actions (only allow hold/close)
                if circuit_active:
                    for i in range(18):
                        mask[i] = 0.0

                # Select and execute action
                action = agent.select_action(state, mask, invert=invert_policy)
                # Snapshot raw PnL before step to detect actual closes
                raw_pnl_before = env.pnl_tracker.daily_pnl()
                next_state, reward, done = env.step(ticker, action)
                raw_pnl_after = env.pnl_tracker.daily_pnl()
                raw_pnl_delta = raw_pnl_after - raw_pnl_before

                # Store transition — fee-aware reward makes all transitions useful
                agent.store_transition(state, action, reward, next_state, done)
                if abs(raw_pnl_delta) > 1e-9:
                    epoch_trades += 1

                # Log to CSV
                decoded = decode_action(action)
                if isinstance(decoded, tuple):
                    action_str = f"BUY_{decoded[0].upper()}_{decoded[1]}_AT_{int(decoded[2]*100)}c"
                else:
                    action_str = decoded.upper()
                csv_writer.writerow([
                    epoch + 1,
                    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    ticker, action_str, f"{agent.epsilon():.4f}",
                    env.pnl_tracker.get_position(ticker),
                    f"{reward:.6f}", f"{raw_pnl_after:.6f}",
                    agent.step_count, f"{env._markets[ticker].last_price:.4f}",
                    f"{spot_price:.2f}",
                ])

            # Train the agent
            agent.step_count += 1
            step_count += 1
            loss = agent.train_step()

            # Periodic logging (every 50 steps)
            if step_count % 50 == 0:
                eps = agent.epsilon()
                daily = env.pnl_tracker.daily_pnl()
                loss_str = f"{loss:.6f}" if loss is not None else "warmup"
                log.info(
                    "Epoch %d | Step %d/%d | eps=%.3f | epoch_pnl=$%.4f | trades=%d | loss=%s",
                    epoch + 1, ts_idx + 1, len(timestamps), eps, daily,
                    epoch_trades, loss_str,
                )

            # Checkpoint (based on total step count across all epochs)
            if step_count % cfg.checkpoint_freq == 0:
                ckpt_dir_path = Path(cfg.checkpoint_dir)
                ckpt_dir_path.mkdir(parents=True, exist_ok=True)
                ckpt_path = str(ckpt_dir_path / f"replay_step_{agent.step_count}.pt")
                agent.save_checkpoint(ckpt_path)
                log.info("Checkpoint saved: %s", ckpt_path)

            # Optional delay for visualization
            if speed > 0:
                import time
                time.sleep(speed)

        # Close remaining positions at end of epoch (forced liquidation)
        for ticker in list(env._markets.keys()):
            pos = env.pnl_tracker.get_position(ticker)
            if pos != 0:
                close_action = 19 if pos > 0 else 20  # CLOSE_YES / CLOSE_NO
                env.step(ticker, close_action)
                epoch_trades += 1

        # Epoch summary
        epoch_pnl = env.pnl_tracker.daily_pnl()
        grand_trades += epoch_trades
        grand_pnl += epoch_pnl

        log.info("=" * 60)
        log.info("EPOCH %d/%d COMPLETE", epoch + 1, n_epochs)
        log.info("  Epoch trades:    %d", epoch_trades)
        log.info("  Epoch PnL:       $%.4f", epoch_pnl)
        log.info("  Grand trades:    %d", grand_trades)
        log.info("  Grand PnL:       $%.4f", grand_pnl)
        log.info("  Epsilon:         %.4f", agent.epsilon())
        log.info("  Buffer size:     %d", len(agent._buffer))
        log.info("=" * 60)

    csv_file.close()

    # ── Final summary ───────────────────────────────────────────────
    avg_pnl = grand_pnl / grand_trades if grand_trades > 0 else 0.0
    summary = {
        "total_steps": step_count,
        "total_trades": grand_trades,
        "total_pnl": grand_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "final_epsilon": agent.epsilon(),
        "replay_buffer_size": len(agent._buffer),
        "data_rows": n_rows,
        "tickers": tickers,
        "csv_log": str(csv_path),
        "n_epochs": n_epochs,
    }

    log.info("=" * 60)
    log.info("REPLAY COMPLETE")
    log.info("=" * 60)
    log.info("Epochs:          %d", n_epochs)
    log.info("Total steps:     %d", summary["total_steps"])
    log.info("Total trades:    %d", summary["total_trades"])
    log.info("Total PnL:       $%.4f", summary["total_pnl"])
    log.info("Avg PnL/trade:   $%.4f", summary["avg_pnl_per_trade"])
    log.info("Final epsilon:   %.4f", summary["final_epsilon"])
    log.info("Replay buffer:   %d transitions", summary["replay_buffer_size"])
    log.info("Trade log:       %s", summary["csv_log"])
    log.info("=" * 60)

    # Save final checkpoint
    ckpt_dir_path = Path(cfg.checkpoint_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    final_path = str(ckpt_dir_path / "replay_final.pt")
    agent.save_checkpoint(final_path)
    log.info("Final checkpoint: %s", final_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay RL agent on saved data")
    parser.add_argument(
        "--data", default=_DEFAULT_DATA,
        help="Path to parquet data file (default: %(default)s)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.0,
        help="Seconds between steps, 0 = no delay (default: 0)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to agent checkpoint to resume from",
    )
    parser.add_argument(
        "--max-loss", type=float, default=50.0,
        help="Circuit breaker max daily loss (default: 50)",
    )
    parser.add_argument(
        "--no-market-limit", action="store_true",
        help="Remove max_markets_per_step cap (act on ALL active markets)",
    )
    parser.add_argument(
        "--prioritize-held", action="store_true",
        help="Always include markets with open positions in per-step selection",
    )
    parser.add_argument(
        "--run-name", default="",
        help="Experiment name suffix for checkpoint dir and CSV log isolation",
    )
    parser.add_argument(
        "--invert-policy", action="store_true",
        help="Invert Q-values (argmin instead of argmax) to test signal direction",
    )
    parser.add_argument(
        "--no-fees", action="store_true",
        help="Set maker/taker fees to 0 and spread to 0 (isolate directional signal)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of passes over the training data (default: 1)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["fast_linear", "exponential", "logarithmic",
                 "episode", "action_local", "parameter_noise"],
        help="Exploration strategy (default: legacy linear decay from config)",
    )
    args = parser.parse_args()

    summary = run_replay(
        args.data, speed=args.speed, checkpoint_path=args.checkpoint,
        max_daily_loss=args.max_loss,
        no_market_limit=args.no_market_limit,
        prioritize_held=args.prioritize_held,
        run_name=args.run_name,
        invert_policy=args.invert_policy,
        no_fees=args.no_fees,
        n_epochs=args.epochs,
        strategy_name=args.strategy,
    )
    # Exit 0 if PnL is reasonable, 1 if catastrophic loss
    sys.exit(0 if summary["total_pnl"] > -1000 * summary["n_epochs"] else 1)


if __name__ == "__main__":
    main()
