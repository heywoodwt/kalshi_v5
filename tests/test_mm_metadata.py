from dataclasses import dataclass
import pytest
import tempfile
import polars as pl
from rl_bot.mm_metadata import MarketMetadata, MarketMetadataLoader


def test_market_metadata_creation():
    """Test MarketMetadata dataclass initialization."""
    meta = MarketMetadata(
        ticker="KXBTC-26JUN2509-B64000",
        price_level_structure="deci_cent",
        tick_size_low=0.001,
        tick_size_mid=0.001,
        tick_size_high=0.001,
    )
    assert meta.ticker == "KXBTC-26JUN2509-B64000"
    assert meta.price_level_structure == "deci_cent"
    assert meta.tick_size_low == 0.001


def test_load_metadata_from_parquet():
    """Test loading metadata from parquet file."""
    # Create test parquet file
    test_data = pl.DataFrame({
        "ticker": ["KXBTC-TEST1", "KXBTC-TEST2"],
        "price_level_structure": ["deci_cent", "linear_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)

        # Load metadata
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        metadata = loader.load_metadata(["KXBTC-TEST1", "KXBTC-TEST2"])

        assert "KXBTC-TEST1" in metadata
        assert metadata["KXBTC-TEST1"].price_level_structure == "deci_cent"
        assert metadata["KXBTC-TEST1"].tick_size_mid == 0.001

        assert "KXBTC-TEST2" in metadata
        assert metadata["KXBTC-TEST2"].price_level_structure == "linear_cent"
        assert metadata["KXBTC-TEST2"].tick_size_mid == 0.01


def test_get_valid_tick_size():
    """Test tick size calculation for different market types."""
    test_data = pl.DataFrame({
        "ticker": ["DECI", "LINEAR", "TAPERED"],
        "price_level_structure": ["deci_cent", "linear_cent", "tapered_deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["DECI", "LINEAR", "TAPERED"])

        # Deci_cent: always 0.001
        assert loader.get_valid_tick_size("DECI", 0.05) == 0.001
        assert loader.get_valid_tick_size("DECI", 0.50) == 0.001
        assert loader.get_valid_tick_size("DECI", 0.95) == 0.001

        # Linear_cent: always 0.01
        assert loader.get_valid_tick_size("LINEAR", 0.05) == 0.01
        assert loader.get_valid_tick_size("LINEAR", 0.50) == 0.01

        # Tapered: 0.001 at tails, 0.01 in middle
        assert loader.get_valid_tick_size("TAPERED", 0.05) == 0.001  # Below 0.10
        assert loader.get_valid_tick_size("TAPERED", 0.50) == 0.01   # 0.10-0.90
        assert loader.get_valid_tick_size("TAPERED", 0.95) == 0.001  # Above 0.90


def test_supports_subpenny():
    """Test subpenny support check."""
    test_data = pl.DataFrame({
        "ticker": ["DECI", "LINEAR", "TAPERED"],
        "price_level_structure": ["deci_cent", "linear_cent", "tapered_deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["DECI", "LINEAR", "TAPERED"])

        # Deci_cent: always supports subpenny
        assert loader.supports_subpenny("DECI", 0.50) is True

        # Linear_cent: never supports subpenny
        assert loader.supports_subpenny("LINEAR", 0.50) is False

        # Tapered: only at tails
        assert loader.supports_subpenny("TAPERED", 0.05) is True   # Below 0.10
        assert loader.supports_subpenny("TAPERED", 0.50) is False  # Middle
        assert loader.supports_subpenny("TAPERED", 0.95) is True   # Above 0.90


def test_missing_ticker():
    """Test behavior with missing ticker."""
    test_data = pl.DataFrame({
        "ticker": ["EXISTING"],
        "price_level_structure": ["deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["EXISTING"])

        # Unknown ticker should default to deci_cent (conservative)
        assert loader.get_valid_tick_size("UNKNOWN", 0.50) == 0.001
        assert loader.supports_subpenny("UNKNOWN", 0.50) is True


def test_parquet_path_validation():
    """Test that parquet path is required."""
    loader = MarketMetadataLoader(mode="parquet", parquet_path=None)
    with pytest.raises(ValueError, match="parquet_path required"):
        loader.load_metadata(["ANY"])


def test_tapered_boundary_prices():
    """Test tapered market at exact boundary prices."""
    test_data = pl.DataFrame({
        "ticker": ["TAPERED"],
        "price_level_structure": ["tapered_deci_cent"],
    })

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        test_data.write_parquet(f.name)
        loader = MarketMetadataLoader(mode="parquet", parquet_path=f.name)
        loader.load_metadata(["TAPERED"])

        # Exact boundary prices
        assert loader.get_valid_tick_size("TAPERED", 0.10) == 0.01  # At boundary, in middle
        assert loader.get_valid_tick_size("TAPERED", 0.90) == 0.01  # At boundary, in middle
        assert loader.get_valid_tick_size("TAPERED", 0.09) == 0.001  # Just below
        assert loader.get_valid_tick_size("TAPERED", 0.91) == 0.001  # Just above