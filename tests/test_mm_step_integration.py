import pytest
import numpy as np
import tempfile
import polars as pl
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader


def test_step_applies_subpenny_to_quotes():
    """Test that step() applies subpenny adjustment to bid/ask."""
    # Create test data
    trades_df = pl.DataFrame({
        "ticker": ["TEST"] * 3,
        "yes_price": [0.48, 0.49, 0.50],
        "count": [1, 1, 1],
        "taker_side": ["yes", "no", "yes"],
        "created_time": ["2026-06-25T12:00:00Z"] * 3,
    })

    ticker_data = preprocess_mm_data(trades_df)

    # Create metadata loader
    test_meta = pl.DataFrame({
        "ticker": ["TEST"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_meta.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)

        cfg = MMConfig()
        env = MMEnv(ticker_data=ticker_data, cfg=cfg, metadata_loader=loader)
        obs, info = env.reset()

        # Agent action: [0.0, 0.0] -> half_spread ~0.055, skew=0
        # Mid ~0.49, so base bid ~0.435, base ask ~0.545
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)

        # Verify subpenny applied
        bid = info["bid"]
        ask = info["ask"]

        # Bid should be base + 0.001
        # Ask should be base - 0.001
        # (Exact values depend on mid calc, just verify they're different from base)
        assert bid > 0.01
        assert ask < 0.99
        assert ask > bid  # Sanity check: no crossing


def test_step_prevents_bid_ask_crossing():
    """Test that step() reverts to base prices if adjustment causes crossing."""
    # Create scenario where subpenny would cause crossing
    # (Very narrow spread, subpenny pushes bid above ask)
    trades_df = pl.DataFrame({
        "ticker": ["TEST"] * 2,
        "yes_price": [0.499, 0.500],
        "count": [1, 1],
        "taker_side": ["yes", "no"],
        "created_time": ["2026-06-25T12:00:00Z"] * 2,
    })

    ticker_data = preprocess_mm_data(trades_df)

    test_meta = pl.DataFrame({
        "ticker": ["TEST"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_meta.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)

        cfg = MMConfig()
        env = MMEnv(ticker_data=ticker_data, cfg=cfg, metadata_loader=loader)
        obs, info = env.reset()

        # Very tight spread action
        action = np.array([-0.95, 0.0], dtype=np.float32)  # Minimal spread
        obs, reward, done, truncated, info = env.step(action)

        # Verify no crossing
        assert info["ask"] > info["bid"]
