import pytest


def test_btc_poller_initial_values():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller(poll_interval_s=30.0)
    assert poller.spot_price() == 0.0
    assert poller.return_5m() == 0.0
    assert poller.return_1h() == 0.0
    assert poller.funding_rate() == 0.0
    assert poller.session_start_price() == 0.0


def test_btc_poller_update_spot():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller(poll_interval_s=30.0)
    # Simulate receiving spot price updates
    poller._on_spot_update(100_000.0)
    assert poller.spot_price() == 100_000.0
    assert poller.session_start_price() == 100_000.0


def test_btc_poller_return_5m():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller(poll_interval_s=30.0)
    # 5m = 10 samples at 30s intervals
    # Push 11 prices so we have enough for a 5m return
    for i in range(11):
        poller._on_spot_update(100_000.0 + i * 100.0)
    ret = poller.return_5m()
    # return = (latest - price_10_samples_ago) / price_10_samples_ago
    # latest = 100_000 + 10*100 = 101_000
    # 10 ago = 100_000
    # return = 1000 / 100_000 = 0.01
    assert abs(ret - 0.01) < 1e-9


def test_btc_poller_return_1h():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller(poll_interval_s=30.0)
    # 1h = 120 samples at 30s intervals
    for i in range(121):
        poller._on_spot_update(100_000.0 + i * 10.0)
    ret = poller.return_1h()
    # latest = 100_000 + 120*10 = 101_200
    # 120 ago = 100_000
    # return = 1200 / 100_000 = 0.012
    assert abs(ret - 0.012) < 1e-9


def test_btc_poller_funding_rate_update():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller(poll_interval_s=30.0)
    poller._on_funding_update(0.0001)
    assert poller.funding_rate() == 0.0001


def test_btc_poller_insufficient_data_returns_zero():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller(poll_interval_s=30.0)
    # Only 3 prices, not enough for 5m return
    for i in range(3):
        poller._on_spot_update(100_000.0 + i * 100.0)
    assert poller.return_5m() == 0.0
    assert poller.return_1h() == 0.0
