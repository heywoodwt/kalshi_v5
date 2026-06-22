import pytest


def test_compute_maker_fee_at_50():
    from rl_bot.reward import compute_maker_fee
    # 1 contract at 0.50 => 0.0175 * 1 * 0.5 * 0.5 = 0.004375 => ceil to 0.01
    fee = compute_maker_fee(1, 0.50, 0.0175)
    assert fee == 0.01


def test_compute_maker_fee_at_extreme():
    from rl_bot.reward import compute_maker_fee
    # 1 contract at 0.95 => 0.0175 * 1 * 0.95 * 0.05 = 0.00083125 => ceil to 0.01
    fee = compute_maker_fee(1, 0.95, 0.0175)
    assert fee == 0.01


def test_compute_maker_fee_multiple_contracts():
    from rl_bot.reward import compute_maker_fee
    # 5 contracts at 0.50 => 0.0175 * 5 * 0.5 * 0.5 = 0.021875 => ceil to 0.03
    fee = compute_maker_fee(5, 0.50, 0.0175)
    assert fee == 0.03


def test_open_and_get_position():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 3, 0.40)
    assert tracker.get_position("KXBTC-A") == 3


def test_open_no_position():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "no", 2, 0.60)
    # "no" direction => position is negative
    assert tracker.get_position("KXBTC-A") == -2


def test_unknown_ticker_position():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    assert tracker.get_position("KXBTC-UNKNOWN") == 0


def test_close_yes_profitable():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    # Buy 1 YES at 0.40, close at 0.60
    tracker.open_position("KXBTC-A", "yes", 1, 0.40)
    pnl = tracker.close_position("KXBTC-A", close_price=0.60)
    # Gross PnL = (0.60 - 0.40) * 1 = 0.20
    # Entry fee = ceil(0.0175 * 1 * 0.40 * 0.60 * 100) / 100 = ceil(0.42) / 100 = 0.01
    # Exit fee  = ceil(0.0175 * 1 * 0.60 * 0.40 * 100) / 100 = ceil(0.42) / 100 = 0.01
    # Net = 0.20 - 0.01 - 0.01 = 0.18
    assert abs(pnl - 0.18) < 1e-9
    assert tracker.get_position("KXBTC-A") == 0


def test_close_no_profitable():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    # Buy 1 NO at 0.60 (pay 1 - 0.60 = 0.40 for NO side)
    # Close when market price drops to 0.40 (NO side now worth 0.60)
    tracker.open_position("KXBTC-A", "no", 1, 0.60)
    pnl = tracker.close_position("KXBTC-A", close_price=0.40)
    # Gross PnL for NO = (entry_yes_price - close_yes_price) * size = (0.60 - 0.40) * 1 = 0.20
    # Entry fee = ceil(0.0175 * 1 * 0.60 * 0.40 * 100) / 100 = 0.01
    # Exit fee  = ceil(0.0175 * 1 * 0.40 * 0.60 * 100) / 100 = 0.01
    # Net = 0.20 - 0.01 - 0.01 = 0.18
    assert abs(pnl - 0.18) < 1e-9


def test_settle_yes_wins():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 1, 0.40)
    pnl = tracker.settle("KXBTC-A", outcome=True)
    # YES settles at 1.00, bought at 0.40
    # Gross = (1.00 - 0.40) * 1 = 0.60
    # Entry fee = ceil(0.0175 * 1 * 0.40 * 0.60 * 100) / 100 = 0.01
    # No exit fee on settlement
    # Net = 0.60 - 0.01 = 0.59
    assert abs(pnl - 0.59) < 1e-9


def test_settle_yes_loses():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 1, 0.40)
    pnl = tracker.settle("KXBTC-A", outcome=False)
    # YES settles at 0.00, bought at 0.40
    # Gross = (0.00 - 0.40) * 1 = -0.40
    # Entry fee = 0.01
    # Net = -0.40 - 0.01 = -0.41
    assert abs(pnl - (-0.41)) < 1e-9


def test_unrealized_pnl():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 2, 0.40)
    upnl = tracker.get_unrealized_pnl("KXBTC-A", market_price=0.50)
    # (0.50 - 0.40) * 2 = 0.20
    assert abs(upnl - 0.20) < 1e-9


def test_total_exposure():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 1, 0.40)
    tracker.open_position("KXBTC-B", "no", 2, 0.60)
    assert tracker.total_exposure() == 2


def test_daily_pnl_accumulates():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 1, 0.40)
    pnl1 = tracker.close_position("KXBTC-A", close_price=0.60)
    tracker.open_position("KXBTC-B", "yes", 1, 0.30)
    pnl2 = tracker.close_position("KXBTC-B", close_price=0.50)
    assert abs(tracker.daily_pnl() - (pnl1 + pnl2)) < 1e-9


def test_daily_pnl_reset():
    from rl_bot.reward import PnLTracker
    tracker = PnLTracker(maker_fee_rate=0.0175)
    tracker.open_position("KXBTC-A", "yes", 1, 0.40)
    tracker.close_position("KXBTC-A", close_price=0.60)
    tracker.reset_daily()
    assert tracker.daily_pnl() == 0.0
