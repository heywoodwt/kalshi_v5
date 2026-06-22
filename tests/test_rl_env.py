import numpy as np
import pytest


def _make_env():
    from rl_bot.config import RLConfig
    from rl_bot.btc_data import BTCDataPoller
    from rl_bot.environment import TradingEnv
    poller = BTCDataPoller()
    # Seed BTC data
    for i in range(121):
        poller._on_spot_update(100_000.0 + i * 10.0)
    poller._on_funding_update(0.0001)
    cfg = RLConfig()
    return TradingEnv(cfg, poller)


def test_env_no_active_markets_initially():
    env = _make_env()
    assert env.get_active_markets() == []


def test_env_market_becomes_active():
    env = _make_env()
    # Need at least 2 price observations for volatility
    env.on_ticker("KXBTC-A", 0.50, 2.0)
    env.on_ticker("KXBTC-A", 0.51, 2.0)
    env.on_ticker("KXBTC-A", 0.52, 2.0)
    assert "KXBTC-A" in env.get_active_markets()


def test_env_get_state_shape():
    env = _make_env()
    env.on_ticker("KXBTC-A", 0.50, 2.0)
    env.on_ticker("KXBTC-A", 0.51, 2.0)
    state = env.get_state("KXBTC-A")
    assert state.shape == (18,)
    assert state.dtype == np.float32


def test_env_step_hold():
    from rl_bot.config import ACTION_HOLD
    env = _make_env()
    env.on_ticker("KXBTC-A", 0.50, 2.0)
    env.on_ticker("KXBTC-A", 0.51, 2.0)
    next_state, reward, done = env.step("KXBTC-A", ACTION_HOLD)
    assert next_state.shape == (18,)
    assert reward == 0.0
    assert done is False


def test_env_step_buy_yes():
    env = _make_env()
    env.on_ticker("KXBTC-A", 0.50, 2.0)
    env.on_ticker("KXBTC-A", 0.51, 2.0)
    # Action 0 = BUY_YES, 1 contract, 0c offset
    next_state, reward, done = env.step("KXBTC-A", 0)
    assert reward == 0.0  # Opening position, no realized PnL
    assert done is False


def test_env_step_close_position():
    from rl_bot.config import ACTION_CLOSE_YES
    env = _make_env()
    env.on_ticker("KXBTC-A", 0.40, 2.0)
    env.on_ticker("KXBTC-A", 0.40, 2.0)
    # Buy 1 YES at 0.40
    env.step("KXBTC-A", 0)
    # Price moves up
    env.on_ticker("KXBTC-A", 0.60, 1.5)
    # Close position
    next_state, reward, done = env.step("KXBTC-A", ACTION_CLOSE_YES)
    # Should have positive realized PnL
    assert reward > 0.0
    assert done is False


def test_env_settle_market():
    env = _make_env()
    env.on_ticker("KXBTC-A", 0.40, 2.0)
    env.on_ticker("KXBTC-A", 0.40, 2.0)
    # Buy 1 YES at 0.40
    env.step("KXBTC-A", 0)
    # Settle — YES wins
    pnl = env.settle_market("KXBTC-A", outcome=True)
    assert pnl > 0.0


def test_env_circuit_breaker():
    env = _make_env()
    assert env.is_circuit_breaker_active() is False


def test_env_mask_after_buy():
    from rl_bot.config import ACTION_CLOSE_YES
    env = _make_env()
    env.on_ticker("KXBTC-A", 0.50, 2.0)
    env.on_ticker("KXBTC-A", 0.50, 2.0)
    # Buy YES
    env.step("KXBTC-A", 0)
    mask = env.get_mask("KXBTC-A")
    # CLOSE_YES should now be valid
    assert mask[ACTION_CLOSE_YES] == 1.0
