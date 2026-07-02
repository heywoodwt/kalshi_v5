import pytest
import numpy as np
import tempfile
import polars as pl
from datetime import datetime, timezone
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader
from model.hp_dfm_rte.orderbook import OrderbookSnapshot


def test_observation_space_shape():
    """Test observation space is 20 dimensions."""
    cfg = MMConfig()
    env = MMEnv(ticker_data={}, cfg=cfg)

    assert env.observation_space.shape == (20,)
    assert env.observation_space.low.shape == (20,)
    assert env.observation_space.high.shape == (20,)


def test_build_obs_with_orderbook():
    """Test building observation with orderbook features."""
    # Create test data with orderbook
    trades_df = pl.DataFrame({
        "ticker": ["TEST"] * 3,
        "yes_price": [0.48, 0.49, 0.50],
        "count": [1, 2, 1],
        "taker_side": ["yes", "no", "yes"],
        "created_time": ["2026-06-25T12:00:00Z"] * 3,
    })

    orderbooks_df = pl.DataFrame({
        "ticker": ["TEST"],
        "yes_best_bid": [0.48],
        "yes_best_size": [10],
        "no_best_bid": [0.50],
        "no_best_size": [8],
        "fetched_at": ["2026-06-25T12:00:00Z"],
    })

    # Preprocess data
    ticker_data = preprocess_mm_data(trades_df, orderbooks_df)

    # Create environment
    cfg = MMConfig()
    env = MMEnv(ticker_data=ticker_data, cfg=cfg)
    obs, info = env.reset()

    # Verify observation shape and features
    assert obs.shape == (20,)
    assert 0.01 <= obs[0] <= 0.99  # mid_price
    assert 0.01 <= obs[1] <= 0.50  # spread
    assert 0.0 <= obs[2] <= 1.0    # bid_depth_l0
    assert 0.0 <= obs[3] <= 1.0    # ask_depth_l0
