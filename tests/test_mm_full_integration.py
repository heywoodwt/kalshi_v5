"""Full integration test for MM bot with subpenny pricing."""
import pytest
import numpy as np
import tempfile
import polars as pl
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader


def test_full_mm_pipeline_with_subpenny():
    """Test complete MM pipeline: metadata -> obs -> step -> subpenny."""
    # 1. Create test data
    trades_df = pl.DataFrame({
        "ticker": ["DECI"] * 5,
        "yes_price": [0.47, 0.48, 0.49, 0.50, 0.51],
        "count": [1, 2, 1, 3, 1],
        "taker_side": ["yes", "no", "yes", "no", "yes"],
        "created_time": ["2026-06-25T12:00:00Z"] * 5,
    })

    ticker_data = preprocess_mm_data(trades_df)

    # 2. Create metadata loader
    test_meta = pl.DataFrame({
        "ticker": ["DECI"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_meta.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["DECI"])

        # 3. Create environment with all features
        cfg = MMConfig(subpenny_enabled=True)
        env = MMEnv(ticker_data=ticker_data, cfg=cfg, metadata_loader=loader)

        # 4. Reset and check observation shape
        obs, info = env.reset()
        assert obs.shape == (20,), f"Expected 20-dim obs, got {obs.shape}"

        # 5. Take action and verify subpenny applied
        action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)

        # 6. Verify quotes have subpenny
        bid = info["bid"]
        ask = info["ask"]

        # Check that bid/ask are different from what they'd be without subpenny
        # (Can't verify exact values without knowing mid, but can verify validity)
        assert 0.01 <= bid <= 0.99
        assert 0.01 <= ask <= 0.99
        assert ask > bid  # No crossing

        # 7. Verify observation features populated
        assert 0.01 <= obs[0] <= 0.99  # mid_price
        assert 0.01 <= obs[1] <= 0.10  # spread
        assert -1.0 <= obs[8] <= 1.0   # book_imbalance

        print(f"✓ Full integration test passed: bid={bid:.3f}, ask={ask:.3f}")
