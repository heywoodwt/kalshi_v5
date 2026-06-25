"""Smoke tests for MM training script.

Verifies that training runs without crashes on synthetic data.
Uses small timestep count (~100) for fast testing.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

# Guard SB3 import for environments without it
sb3 = pytest.importorskip("stable_baselines3")

from rl_bot.mm_config import MMConfig
from rl_bot.mm_train import list_categories, train_category


def create_synthetic_trade_data(n_trades: int = 100) -> pl.DataFrame:
    """Generate synthetic trade data for testing.

    Creates realistic Kalshi BTC trade data with multiple tickers.

    Args:
        n_trades: Number of trades to generate per ticker

    Returns:
        Polars DataFrame with columns: ticker, yes_price, count, taker_side, created_time
    """
    # Generate 2 tickers in KXBTC15M category
    tickers = ["KXBTC15M-26JUN24-T17", "KXBTC15M-26JUN24-T18"]
    data = []

    base_time = datetime(2024, 6, 26, 15, 0, 0)

    for ticker in tickers:
        # Generate n_trades for each ticker
        for i in range(n_trades):
            # Random walk around 0.50
            price = 0.50 + np.random.normal(0, 0.02)
            price = max(0.01, min(0.99, price))

            # Random trade size (1-10 contracts)
            count = np.random.randint(1, 11)

            # Random taker side
            taker_side = np.random.choice(["yes", "no"])

            # Increment time by 1 minute per trade
            trade_time = base_time + timedelta(minutes=i)

            data.append({
                "ticker": ticker,
                "yes_price": price,
                "count": count,
                "taker_side": taker_side,
                "created_time": trade_time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            })

    return pl.DataFrame(data)


def test_list_categories():
    """Test category extraction from ticker names."""
    df = pl.DataFrame({
        "ticker": [
            "KXBTC15M-26JUN24-T17",
            "KXBTC15M-26JUN24-T18",
            "KXBTC1H-26JUN24-T17",
            "INXD-26JUN24-100",
        ]
    })

    categories = list_categories(df)

    # Should extract KXBTC, INXD
    assert "KXBTC" in categories or "KXBTCM" in categories or "KXBTCH" in categories
    assert "INXD" in categories
    assert len(categories) >= 2


def test_train_category_smoke():
    """Smoke test: train for 100 steps without crashing."""
    # Create synthetic data
    df = create_synthetic_trade_data(n_trades=100)

    # Use minimal config for fast testing
    cfg = MMConfig(
        total_timesteps=100,  # Very short training
        n_steps=32,  # Small rollout buffer
        batch_size=16,  # Small batch
        min_trades_per_ticker=50,  # Low threshold
    )

    # Train on KXBTC category
    checkpoint_path = train_category(
        df,
        category="KXBTC",
        cfg=cfg,
        run_name="test_run",
    )

    # Verify checkpoint was created
    if checkpoint_path:  # May be empty if no tickers meet threshold
        from pathlib import Path
        assert Path(checkpoint_path).exists()
        assert checkpoint_path.endswith(".zip")


def test_train_no_liquid_tickers():
    """Test handling when no tickers meet liquidity threshold."""
    # Create data with very few trades
    df = create_synthetic_trade_data(n_trades=10)

    cfg = MMConfig(
        total_timesteps=100,
        min_trades_per_ticker=50,  # Threshold higher than data
    )

    # Should handle gracefully (return empty string)
    checkpoint_path = train_category(
        df,
        category="KXBTC",
        cfg=cfg,
        run_name="test_run",
    )

    assert checkpoint_path == ""
