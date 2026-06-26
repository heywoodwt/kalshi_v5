import math
from datetime import datetime, timedelta, timezone

import pytest

from model.hp_dfm_rte.orderbook import OrderbookSnapshot


# ── estimate_spread ──────────────────────────────────────────────────

def test_spread_at_midprice():
    """Spread at p=0.50 should equal base_spread (all factors ~1.0)."""
    from rl_bot.execution import estimate_spread
    # p=0.50 -> variance=0.25, price_factor=1.0
    # tte=0 -> tte_factor=1.0
    # volume=0 -> vol_factor=1.0
    s = estimate_spread(0.50, 0.0, 0.0, base_spread=0.03)
    assert abs(s - 0.03) < 1e-6


def test_spread_wider_at_extremes():
    """Spread should be wider at p=0.95 than p=0.50."""
    from rl_bot.execution import estimate_spread
    s_mid = estimate_spread(0.50, 1.0, 5.0)
    s_extreme = estimate_spread(0.95, 1.0, 5.0)
    assert s_extreme > s_mid


def test_spread_tighter_with_volume():
    """More volume should produce a tighter spread."""
    from rl_bot.execution import estimate_spread
    s_quiet = estimate_spread(0.50, 1.0, 0.0)
    s_busy = estimate_spread(0.50, 1.0, 50.0)
    assert s_busy < s_quiet


def test_spread_clamped_low():
    """Spread should never go below 0.01."""
    from rl_bot.execution import estimate_spread
    # Huge volume, short TTE, mid-price -> spread gets very small
    s = estimate_spread(0.50, 0.0, 1000.0, base_spread=0.01)
    assert s >= 0.01


def test_spread_clamped_high():
    """Spread should never exceed 0.10."""
    from rl_bot.execution import estimate_spread
    # Extreme price, far from expiry, no volume
    s = estimate_spread(0.99, 24.0, 0.0, base_spread=0.10)
    assert s <= 0.10


# ── compute_slippage ─────────────────────────────────────────────────

def test_slippage_zero_for_small_order():
    """No slippage when order fits in top-of-book."""
    from rl_bot.execution import compute_slippage
    assert compute_slippage(5, top_of_book_size=10) == 0.0


def test_slippage_one_level():
    """Walking one level beyond top-of-book costs 1 cent."""
    from rl_bot.execution import compute_slippage
    # 15 contracts, 10 at top -> 5 leftover, ceil(5/10)=1 level
    assert compute_slippage(15, top_of_book_size=10) == 0.01


def test_slippage_two_levels():
    """Walking two levels costs 2 cents."""
    from rl_bot.execution import compute_slippage
    # 25 contracts, 10 at top -> 15 leftover, ceil(15/10)=2 levels
    assert compute_slippage(25, top_of_book_size=10) == 0.02


def test_slippage_exact_boundary():
    """Exactly at top-of-book boundary -> no slippage."""
    from rl_bot.execution import compute_slippage
    assert compute_slippage(10, top_of_book_size=10) == 0.0


# ── compute_fill_probability ─────────────────────────────────────────

def test_fill_prob_market_order():
    """Market orders (offset=0) always fill."""
    from rl_bot.execution import compute_fill_probability
    assert compute_fill_probability(0.0, 0.03, 5.0) == 1.0


def test_fill_prob_decreases_with_offset():
    """Deeper limit orders fill less often."""
    from rl_bot.execution import compute_fill_probability
    p_tight = compute_fill_probability(0.01, 0.03, 5.0)
    p_deep = compute_fill_probability(0.04, 0.03, 5.0)
    assert p_tight > p_deep


def test_fill_prob_increases_with_volume():
    """More volume -> higher fill probability for limits."""
    from rl_bot.execution import compute_fill_probability
    p_quiet = compute_fill_probability(0.02, 0.03, 0.0)
    p_busy = compute_fill_probability(0.02, 0.03, 50.0)
    assert p_busy > p_quiet


def test_fill_prob_clamped():
    """Fill probability stays in [0.05, 1.0]."""
    from rl_bot.execution import compute_fill_probability
    # Very deep offset, no volume -> close to minimum
    p = compute_fill_probability(0.10, 0.01, 0.0)
    assert p >= 0.05
    assert p <= 1.0


# ── classify_fee_rate ────────────────────────────────────────────────

def test_fee_market_order_is_taker():
    from rl_bot.execution import classify_fee_rate
    assert classify_fee_rate(0.0, is_close=False) == 0.07


def test_fee_limit_order_is_maker():
    from rl_bot.execution import classify_fee_rate
    assert classify_fee_rate(0.02, is_close=False) == 0.0175


def test_fee_close_always_taker():
    from rl_bot.execution import classify_fee_rate
    # Even with offset, closes are taker
    assert classify_fee_rate(0.02, is_close=True) == 0.07
    assert classify_fee_rate(0.0, is_close=True) == 0.07


# ── compute_execution (integration) ─────────────────────────────────

def test_execution_buy_yes_market():
    """Market buy YES should pay above mid (ask + slippage)."""
    from rl_bot.execution import compute_execution
    r = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0)
    assert r.exec_price > 0.50
    assert r.fee_rate == 0.07       # taker
    assert r.fill_prob == 1.0       # market order


def test_execution_buy_yes_limit():
    """Limit buy YES should have lower price than market buy."""
    from rl_bot.execution import compute_execution
    market = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0)
    limit = compute_execution("yes", 1, 0.02, 0.50, 2.0, 5.0)
    assert limit.exec_price < market.exec_price
    assert limit.fee_rate == 0.0175  # maker
    assert limit.fill_prob < 1.0     # may not fill


def test_execution_buy_no_market():
    """Market buy NO records below mid (bid - slippage)."""
    from rl_bot.execution import compute_execution
    r = compute_execution("no", 1, 0.0, 0.50, 2.0, 5.0)
    assert r.exec_price < 0.50
    assert r.fee_rate == 0.07
    assert r.fill_prob == 1.0


def test_execution_close_yes():
    """Closing YES hits the bid (below mid)."""
    from rl_bot.execution import compute_execution
    r = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0, is_close=True)
    assert r.exec_price < 0.50
    assert r.fee_rate == 0.07  # always taker


def test_execution_close_no():
    """Closing NO hits the ask (above mid)."""
    from rl_bot.execution import compute_execution
    r = compute_execution("no", 1, 0.0, 0.50, 2.0, 5.0, is_close=True)
    assert r.exec_price > 0.50
    assert r.fee_rate == 0.07


def test_execution_price_clamped():
    """Execution price stays in [0.01, 0.99]."""
    from rl_bot.execution import compute_execution
    # Extreme scenario: buy NO at very low price with large slippage
    r = compute_execution("no", 50, 0.0, 0.02, 1.0, 0.0, top_of_book_size=1)
    assert r.exec_price >= 0.01
    # Extreme scenario: buy YES at very high price
    r2 = compute_execution("yes", 50, 0.0, 0.98, 1.0, 0.0, top_of_book_size=1)
    assert r2.exec_price <= 0.99


# ── Orderbook integration ───────────────────────────────────────────

def _fresh_book(
    yes_price: float = 0.48,
    no_price: float = 0.54,
    yes_size: int = 20,
    no_size: int = 15,
    age_seconds: float = 5.0,
) -> OrderbookSnapshot:
    """Build an OrderbookSnapshot that passes the freshness check."""
    return OrderbookSnapshot(
        ticker="TEST-BOOK",
        yes_price=yes_price,
        no_price=no_price,
        yes_size=yes_size,
        no_size=no_size,
        updated_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )


def test_orderbook_fresh_uses_real_spread():
    """When the orderbook is fresh, spread comes from real quotes."""
    from rl_bot.execution import compute_execution, _spread_from_orderbook
    book = _fresh_book(yes_price=0.48, no_price=0.54)
    expected_spread = _spread_from_orderbook(book)
    r = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0, orderbook=book)
    assert r.spread == expected_spread


def test_orderbook_stale_falls_back():
    """An orderbook older than 30 s should fall back to estimation."""
    from rl_bot.execution import compute_execution, estimate_spread
    stale = _fresh_book(age_seconds=60.0)
    r = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0, orderbook=stale)
    expected = estimate_spread(0.50, 2.0, 5.0, 0.03)
    assert r.spread == expected


def test_orderbook_none_falls_back():
    """orderbook=None (default) should fall back to estimation."""
    from rl_bot.execution import compute_execution, estimate_spread
    r = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0, orderbook=None)
    expected = estimate_spread(0.50, 2.0, 5.0, 0.03)
    assert r.spread == expected


def test_orderbook_missing_side_falls_back():
    """An orderbook with only one side quoted should fall back."""
    from rl_bot.execution import compute_execution, estimate_spread
    one_side = OrderbookSnapshot(
        ticker="HALF",
        yes_price=0.48,
        no_price=None,
        updated_at=datetime.now(timezone.utc),
    )
    r = compute_execution("yes", 1, 0.0, 0.50, 2.0, 5.0, orderbook=one_side)
    expected = estimate_spread(0.50, 2.0, 5.0, 0.03)
    assert r.spread == expected


def test_orderbook_depth_open_yes():
    """Opening YES uses no_size (opposite side) for slippage depth."""
    from rl_bot.execution import compute_execution, compute_slippage
    # no_size=5 means only 5 at best ask; 10-contract order walks the book
    book = _fresh_book(no_size=5)
    r = compute_execution("yes", 10, 0.0, 0.50, 2.0, 5.0, orderbook=book)
    expected_slip = compute_slippage(10, max(1, 5))
    assert r.slippage == expected_slip


def test_orderbook_depth_close_yes():
    """Closing YES uses yes_size (same side) for slippage depth."""
    from rl_bot.execution import compute_execution, compute_slippage
    # yes_size=3 means thin bid; large close order gets more slippage
    book = _fresh_book(yes_size=3)
    r = compute_execution(
        "yes", 10, 0.0, 0.50, 2.0, 5.0, is_close=True, orderbook=book,
    )
    expected_slip = compute_slippage(10, max(1, 3))
    assert r.slippage == expected_slip


def test_spread_from_orderbook_helper():
    """_spread_from_orderbook computes yes_ask - yes_price, clamped."""
    from rl_bot.execution import _spread_from_orderbook
    book = _fresh_book(yes_price=0.48, no_price=0.54)
    # yes_ask = 1.0 - 0.54 = 0.46  →  spread = 0.46 - 0.48 = -0.02  → clamped to 0.01
    assert _spread_from_orderbook(book) == 0.01

    # Normal positive spread: yes=0.45, no=0.50 → ask=0.50, spread=0.05
    book2 = _fresh_book(yes_price=0.45, no_price=0.50)
    assert abs(_spread_from_orderbook(book2) - 0.05) < 1e-9


def test_orderbook_is_fresh_helper():
    """_orderbook_is_fresh returns correct booleans for edge cases."""
    from rl_bot.execution import _orderbook_is_fresh
    assert _orderbook_is_fresh(None) is False
    # Empty ticker, no prices
    assert _orderbook_is_fresh(OrderbookSnapshot(ticker="")) is False
    # Fresh and complete
    assert _orderbook_is_fresh(_fresh_book()) is True
    # Stale
    assert _orderbook_is_fresh(_fresh_book(age_seconds=31.0)) is False
