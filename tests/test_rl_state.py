import math
import numpy as np
import pytest


def _make_vol_metrics(**overrides):
    defaults = {
        "vol": 0.01, "range": 0.05, "vol_ratio": 2.0,
        "vol_up": 0.005, "vol_down": 0.005, "momentum": 0.02,
        "last_price": 0.50,
    }
    defaults.update(overrides)
    return defaults


def _make_orderbook():
    from model.hp_dfm_rte.orderbook import OrderbookSnapshot
    return OrderbookSnapshot(ticker="KXBTC-A", yes_price=0.49, yes_size=10, no_price=0.51, no_size=8)


def _make_btc_poller():
    from rl_bot.btc_data import BTCDataPoller
    poller = BTCDataPoller()
    # Seed with enough data
    for i in range(121):
        poller._on_spot_update(100_000.0 + i * 10.0)
    poller._on_funding_update(0.0001)
    return poller


def _make_pnl_tracker():
    from rl_bot.reward import PnLTracker
    return PnLTracker(maker_fee_rate=0.0175)


def test_build_state_shape():
    from rl_bot.config import RLConfig
    from rl_bot.state_builder import build_state
    state = build_state(
        ticker="KXBTC-A",
        vol_metrics=_make_vol_metrics(),
        orderbook=_make_orderbook(),
        btc_poller=_make_btc_poller(),
        pnl_tracker=_make_pnl_tracker(),
        market_price=0.50,
        time_to_expiry_h=2.0,
        trade_count=10,
        cfg=RLConfig(),
    )
    assert state.shape == (18,)
    assert state.dtype == np.float32


def test_build_state_values_reasonable():
    from rl_bot.config import RLConfig
    from rl_bot.state_builder import build_state
    state = build_state(
        ticker="KXBTC-A",
        vol_metrics=_make_vol_metrics(),
        orderbook=_make_orderbook(),
        btc_poller=_make_btc_poller(),
        pnl_tracker=_make_pnl_tracker(),
        market_price=0.50,
        time_to_expiry_h=2.0,
        trade_count=10,
        cfg=RLConfig(),
    )
    # All values should be finite
    assert np.isfinite(state).all()
    # market_price (index 0) should be 0.50
    assert abs(state[0] - 0.50) < 1e-6
    # time_to_expiry (index 6) should be log(1 + 2.0)
    assert abs(state[6] - math.log(1 + 2.0)) < 1e-6


def test_build_state_with_position():
    from rl_bot.config import RLConfig
    from rl_bot.state_builder import build_state
    pnl = _make_pnl_tracker()
    pnl.open_position("KXBTC-A", "yes", 3, 0.40)
    state = build_state(
        ticker="KXBTC-A",
        vol_metrics=_make_vol_metrics(),
        orderbook=_make_orderbook(),
        btc_poller=_make_btc_poller(),
        pnl_tracker=pnl,
        market_price=0.50,
        time_to_expiry_h=2.0,
        trade_count=10,
        cfg=RLConfig(),
    )
    # current_position (index 15) should be 3/5 = 0.6
    assert abs(state[15] - 0.6) < 1e-6


def test_build_action_mask_no_position():
    from rl_bot.config import RLConfig, ACTION_HOLD, ACTION_CLOSE_YES, ACTION_CLOSE_NO
    from rl_bot.state_builder import build_action_mask
    pnl = _make_pnl_tracker()
    mask = build_action_mask("KXBTC-A", pnl, RLConfig())
    assert mask.shape == (21,)
    # All buy actions valid, HOLD valid, CLOSE_YES/CLOSE_NO invalid (no position)
    assert mask[ACTION_HOLD] == 1.0
    assert mask[ACTION_CLOSE_YES] == 0.0
    assert mask[ACTION_CLOSE_NO] == 0.0
    # Buy YES actions valid (indices 0-8)
    assert mask[0] == 1.0
    assert mask[8] == 1.0
    # Buy NO actions valid (indices 9-17)
    assert mask[9] == 1.0
    assert mask[17] == 1.0


def test_build_action_mask_max_yes_position():
    from rl_bot.config import RLConfig, ACTION_HOLD, ACTION_CLOSE_YES, ACTION_CLOSE_NO
    from rl_bot.state_builder import build_action_mask
    pnl = _make_pnl_tracker()
    pnl.open_position("KXBTC-A", "yes", 5, 0.40)
    mask = build_action_mask("KXBTC-A", pnl, RLConfig())
    # All BUY_YES (0-8) should be masked
    for i in range(9):
        assert mask[i] == 0.0
    # BUY_NO (9-17) still valid
    for i in range(9, 18):
        assert mask[i] == 1.0
    # CLOSE_YES valid, CLOSE_NO invalid
    assert mask[ACTION_CLOSE_YES] == 1.0
    assert mask[ACTION_CLOSE_NO] == 0.0


def test_build_action_mask_max_exposure():
    from rl_bot.config import RLConfig, ACTION_HOLD
    from rl_bot.state_builder import build_action_mask
    pnl = _make_pnl_tracker()
    # Fill up 10 markets (max exposure)
    for i in range(10):
        pnl.open_position(f"KXBTC-{i}", "yes", 1, 0.50)
    # 11th market has no position, but exposure is maxed
    mask = build_action_mask("KXBTC-NEW", pnl, RLConfig())
    # All buy actions masked
    for i in range(18):
        assert mask[i] == 0.0
    # HOLD still valid
    assert mask[ACTION_HOLD] == 1.0
