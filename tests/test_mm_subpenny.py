import pytest
import tempfile
import polars as pl
from rl_bot.mm_env import MMEnv
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader


def test_apply_subpenny_bid_adjustment():
    """Test subpenny adjustment adds 0.001 to bid."""
    # Create test metadata
    test_data = pl.DataFrame({
        "ticker": ["DECI"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["DECI"])

        # Create minimal environment
        cfg = MMConfig()
        env = MMEnv(ticker_data={}, cfg=cfg, metadata_loader=loader)
        env._current_ticker = "DECI"

        # Test bid adjustment (add 0.001)
        bid_base = 0.463
        bid_adjusted = env._apply_subpenny_adjustment(bid_base, "bid")
        assert bid_adjusted == pytest.approx(0.464, abs=0.0001)


def test_apply_subpenny_ask_adjustment():
    """Test subpenny adjustment subtracts 0.001 from ask."""
    test_data = pl.DataFrame({
        "ticker": ["DECI"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["DECI"])

        cfg = MMConfig()
        env = MMEnv(ticker_data={}, cfg=cfg, metadata_loader=loader)
        env._current_ticker = "DECI"

        # Test ask adjustment (subtract 0.001)
        ask_base = 0.537
        ask_adjusted = env._apply_subpenny_adjustment(ask_base, "ask")
        assert ask_adjusted == pytest.approx(0.536, abs=0.0001)


def test_no_subpenny_on_linear_cent():
    """Test no adjustment on linear_cent markets."""
    test_data = pl.DataFrame({
        "ticker": ["LINEAR"],
        "price_level_structure": ["linear_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["LINEAR"])

        cfg = MMConfig()
        env = MMEnv(ticker_data={}, cfg=cfg, metadata_loader=loader)
        env._current_ticker = "LINEAR"

        # No adjustment on linear_cent
        bid_base = 0.463
        bid_adjusted = env._apply_subpenny_adjustment(bid_base, "bid")
        assert bid_adjusted == pytest.approx(bid_base, abs=0.0001)


def test_subpenny_price_clamping():
    """Test price clamping at boundaries."""
    test_data = pl.DataFrame({
        "ticker": ["DECI"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["DECI"])

        cfg = MMConfig()
        env = MMEnv(ticker_data={}, cfg=cfg, metadata_loader=loader)
        env._current_ticker = "DECI"

        # Test ask near 0.01 boundary
        ask_low = 0.011
        ask_adjusted = env._apply_subpenny_adjustment(ask_low, "ask")
        assert ask_adjusted >= 0.01  # Should clamp to minimum

        # Test bid near 0.99 boundary
        bid_high = 0.989
        bid_adjusted = env._apply_subpenny_adjustment(bid_high, "bid")
        assert bid_adjusted <= 0.99  # Should clamp to maximum
