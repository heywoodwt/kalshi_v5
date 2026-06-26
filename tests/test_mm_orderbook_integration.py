import pytest
from datetime import datetime, timezone
from model.hp_dfm_rte.orderbook import OrderbookSnapshot


def test_orderbook_mid_price():
    """Test mid price calculation from best bid/ask."""
    ob = OrderbookSnapshot(
        ticker="TEST",
        yes_price=0.48,  # Best yes bid
        yes_size=10,
        no_price=0.50,   # Best no bid (implies yes ask = 0.50)
        no_size=8,
    )

    # Mid = (yes_bid + yes_ask) / 2 = (0.48 + 0.50) / 2 = 0.49
    assert ob.mid_price() == pytest.approx(0.49, abs=0.001)

    # Test None handling
    ob_no_price = OrderbookSnapshot(
        ticker="TEST",
        yes_price=None,
        yes_size=0,
        no_price=0.50,
        no_size=8,
    )
    assert ob_no_price.mid_price() is None


def test_orderbook_spread():
    """Test spread calculation."""
    ob = OrderbookSnapshot(
        ticker="TEST",
        yes_price=0.48,
        yes_size=10,
        no_price=0.50,  # Yes ask = 1.0 - 0.50 = 0.50
        no_size=8,
    )

    # Spread = yes_ask - yes_bid = 0.50 - 0.48 = 0.02
    assert ob.spread() == pytest.approx(0.02, abs=0.001)


def test_orderbook_imbalance():
    """Test book imbalance calculation."""
    ob = OrderbookSnapshot(
        ticker="TEST",
        yes_price=0.48,
        yes_size=10,   # More bid size
        no_price=0.50,
        no_size=8,
    )

    # Imbalance = (10 - 8) / (10 + 8) = 2/18 = 0.111
    assert ob.imbalance() == pytest.approx(0.111, abs=0.01)

    # Test zero total size
    ob_zero = OrderbookSnapshot(
        ticker="TEST",
        yes_price=0.48,
        yes_size=0,
        no_price=0.50,
        no_size=0,
    )
    assert ob_zero.imbalance() == 0.0


def test_orderbook_multilevel_depth():
    """Test orderbook with 3-level depth."""
    ob = OrderbookSnapshot(
        ticker="TEST",
        # Level 0 (best)
        yes_price=0.48,
        yes_size=10,
        no_price=0.50,
        no_size=8,
        # Level 1
        yes_price_l1=0.47,
        yes_size_l1=15,
        no_price_l1=0.51,
        no_size_l1=12,
        # Level 2
        yes_price_l2=0.46,
        yes_size_l2=20,
        no_price_l2=0.52,
        no_size_l2=18,
    )

    # Verify all levels stored correctly
    assert ob.yes_price_l1 == 0.47
    assert ob.yes_size_l1 == 15
    assert ob.no_price_l2 == 0.52
    assert ob.no_size_l2 == 18