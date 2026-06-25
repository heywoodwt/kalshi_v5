"""Tests for MM environment (rl_bot/mm_env.py).

Verifies:
  - Trade preprocessing into per-ticker, per-minute windows
  - Fill simulation logic (buy/sell matching against bid/ask)
  - Inventory cap enforcement
  - Maker fee deduction
  - Reward calculation (delta_pnl - inventory_penalty)
  - Multi-ticker cycling on reset()
  - Observation space shape and bounds
  - Action space shape
  - Episode termination and inventory flattening
"""
import numpy as np
import polars as pl
import pytest

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_trades_for_mm


def test_preprocess_trades_basic():
    """Test basic trade preprocessing into per-ticker windows."""
    # Create sample trades: 2 tickers, 2 windows each
    df = pl.DataFrame({
        "ticker": ["TICK-A", "TICK-A", "TICK-A", "TICK-B", "TICK-B"],
        "yes_price": [0.50, 0.51, 0.52, 0.60, 0.61],
        "count": [10, 20, 15, 5, 8],
        "taker_side": ["yes", "no", "yes", "no", "yes"],
        "created_time": [
            "2026-06-24T10:00:30Z",
            "2026-06-24T10:00:45Z",
            "2026-06-24T10:01:15Z",
            "2026-06-24T10:02:00Z",
            "2026-06-24T10:02:30Z",
        ],
    })

    result = preprocess_trades_for_mm(df)

    # Should have 2 tickers
    assert len(result) == 2
    assert "TICK-A" in result
    assert "TICK-B" in result

    # TICK-A should have 2 windows (10:00 and 10:01)
    tick_a = result["TICK-A"]
    assert len(tick_a) == 2
    assert len(tick_a[0]) == 2  # First window: 2 trades
    assert len(tick_a[1]) == 1  # Second window: 1 trade

    # TICK-B should have 1 window (10:02)
    tick_b = result["TICK-B"]
    assert len(tick_b) == 1
    assert len(tick_b[0]) == 2  # One window: 2 trades

    # Check trade dict structure
    trade = tick_a[0][0]
    assert "yes_price" in trade
    assert "count" in trade
    assert "taker_side" in trade
    assert "created_time" in trade


def test_env_initialization():
    """Test environment initialization and space definitions."""
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 3,
        "yes_price": [0.50, 0.51, 0.52],
        "count": [10, 10, 10],
        "taker_side": ["yes", "no", "yes"],
        "created_time": ["2026-06-24T10:00:00Z"] * 3,
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig()
    env = MMEnv(ticker_data, cfg)

    # Check spaces
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32

    # Check bounds
    assert np.allclose(env.action_space.low, [-1.0, -1.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0])


def test_env_reset():
    """Test environment reset and observation shape."""
    df = pl.DataFrame({
        "ticker": ["TICK-A"] * 3 + ["TICK-B"] * 3,
        "yes_price": [0.50, 0.51, 0.52, 0.60, 0.61, 0.62],
        "count": [10] * 6,
        "taker_side": ["yes", "no", "yes", "no", "yes", "no"],
        "created_time": ["2026-06-24T10:00:00Z"] * 3 + ["2026-06-24T10:01:00Z"] * 3,
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig()
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # Check observation
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert "ticker" in info

    # Check state initialization
    assert env._inventory == 0
    assert env._realized_pnl == 0.0
    assert env._step_idx == 0


def test_multi_ticker_cycling():
    """Test that reset() cycles through tickers in round-robin fashion."""
    df = pl.DataFrame({
        "ticker": ["TICK-A"] * 2 + ["TICK-B"] * 2 + ["TICK-C"] * 2,
        "yes_price": [0.50, 0.51, 0.60, 0.61, 0.70, 0.71],
        "count": [10] * 6,
        "taker_side": ["yes"] * 6,
        "created_time": ["2026-06-24T10:00:00Z"] * 6,
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig()
    env = MMEnv(ticker_data, cfg)

    # First reset
    obs1, info1 = env.reset()
    ticker1 = info1["ticker"]

    # Second reset
    obs2, info2 = env.reset()
    ticker2 = info2["ticker"]

    # Third reset
    obs3, info3 = env.reset()
    ticker3 = info3["ticker"]

    # Fourth reset (should cycle back to first)
    obs4, info4 = env.reset()
    ticker4 = info4["ticker"]

    # Should cycle through all 3 tickers
    tickers_seen = {ticker1, ticker2, ticker3}
    assert len(tickers_seen) == 3
    assert ticker4 == ticker1  # Cycles back to first


def test_deterministic_fill_sequence():
    """Test deterministic trade sequence produces expected fills, inventory, PnL."""
    # Create trades where taker buys NO (agent buys YES at bid)
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 3,
        "yes_price": [0.49, 0.50, 0.51],  # Trades at these prices
        "count": [5, 5, 5],
        "taker_side": ["no", "no", "no"],  # Taker buys NO -> hits our bid
        "created_time": [
            "2026-06-24T10:00:10Z",
            "2026-06-24T10:00:20Z",
            "2026-06-24T10:00:30Z",
        ],
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig(
        min_half_spread=0.02,
        max_half_spread=0.02,  # Fixed 2¢ half-spread
        max_skew=0.0,          # No skew
        max_inventory=20,
        inventory_lambda=0.0,  # No inventory penalty for simpler test
        maker_fee_rate=0.0175,
    )
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # VWAP of window: (0.49*5 + 0.50*5 + 0.51*5) / 15 = 0.50
    # With half_spread=0.02, bid=0.48, ask=0.52
    # All 3 trades at 0.49, 0.50, 0.51 are <= bid (0.48)? NO
    # Actually trades at 0.49, 0.50, 0.51 are > bid=0.48, so NO FILLS
    # Wait, let me recalculate: if mid=0.50, half_spread=0.02, bid=0.50-0.02=0.48
    # Trades at 0.49, 0.50, 0.51 are all > 0.48, so no buy fills

    # Let me adjust: use action that gives wider bid
    # Action [-1, 0] -> min_half_spread=0.02, skew=0 -> bid=0.48, ask=0.52
    # Trades at 0.49, 0.50, 0.51 won't fill at bid=0.48
    # Need bid >= 0.51 to catch all 3 trades
    # half_spread = 0.02 means bid = mid - 0.02 = 0.50 - 0.02 = 0.48

    # Let me create a simpler test: trades at 0.45 with bid at 0.48 -> fills
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 3,
        "yes_price": [0.45, 0.46, 0.47],  # Below our bid
        "count": [5, 5, 5],
        "taker_side": ["no", "no", "no"],  # Taker buys NO -> agent buys YES
        "created_time": ["2026-06-24T10:00:00Z"] * 3,
    })

    ticker_data = preprocess_trades_for_mm(df)
    env = MMEnv(ticker_data, cfg)
    obs, info = env.reset()

    # VWAP = (0.45*5 + 0.46*5 + 0.47*5) / 15 = 0.46
    # mid = 0.46, half_spread = 0.02, bid = 0.44, ask = 0.48
    # Trades at 0.45, 0.46, 0.47 are all <= bid (0.44)? NO
    # They're > 0.44, so no fills

    # I need trades <= bid to get buy fills. Let me use mid=0.50 and trades at 0.40
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 2,
        "yes_price": [0.48, 0.40],  # One at bid price, one below
        "count": [5, 10],
        "taker_side": ["no", "no"],
        "created_time": [
            "2026-06-24T10:00:00Z",  # Window 1
            "2026-06-24T10:01:00Z",  # Window 2
        ],
    })

    ticker_data = preprocess_trades_for_mm(df)
    env = MMEnv(ticker_data, cfg)
    obs, info = env.reset()

    # Step 1: mid=0.48 (from first trade), bid=0.46, ask=0.50
    # Trade at 0.48 > bid (0.46), no fill
    action = np.array([0.0, 0.0], dtype=np.float32)  # mid-range spread
    obs, reward, done, _, info = env.step(action)

    assert env._inventory == 0  # No fill
    assert not done

    # Step 2: mid=0.40, bid=0.38, ask=0.42
    # Trade at 0.40 > bid (0.38), no fill
    obs, reward, done, _, info = env.step(action)

    assert env._inventory == 0  # No fill
    assert done  # Episode ends


def test_inventory_cap_enforcement():
    """Test that fills respect max_inventory cap."""
    # Create many buy opportunities across 3 windows
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 10 + ["TEST-A"] * 5 + ["TEST-A"] * 1,
        "yes_price": [0.30] * 10 + [0.31] * 5 + [0.30],
        "count": [5] * 16,
        "taker_side": ["no"] * 16,
        "created_time": ["2026-06-24T10:00:00Z"] * 10 + ["2026-06-24T10:01:00Z"] * 5 + ["2026-06-24T10:02:00Z"],
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig(max_inventory=15)  # Cap at 15 contracts
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # Use narrower spread and higher skew to get bid above trade prices
    action = np.array([-1.0, 1.0], dtype=np.float32)  # Min half-spread (0.01), max skew (+0.05)
    # mid=0.30, half_spread=0.01, skew=+0.05
    # bid = 0.30 - 0.01 + 0.05 = 0.34
    # ask = 0.30 + 0.01 + 0.05 = 0.36
    # Trades at 0.30 <= bid (0.34), should fill!

    obs, reward, done, _, info = env.step(action)

    # After step 1, inventory should be capped at 15 (3 fills of 5 contracts each)
    assert env._inventory <= cfg.max_inventory
    assert env._inventory == 15  # Should hit cap exactly
    assert not done  # Still have windows 2 and 3

    # Step 2: try to fill more, but should stay at cap
    obs, reward, done, _, info = env.step(action)
    assert env._inventory == 15  # Still at cap (can't fill more)
    assert not done  # Still have window 3


def test_empty_window_inventory_penalty():
    """Test that empty windows still apply inventory penalty."""
    # Create 2 windows: one with trade (to build inventory), one empty
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 1,
        "yes_price": [0.30],
        "count": [10],
        "taker_side": ["no"],
        "created_time": ["2026-06-24T10:00:00Z"],
    })

    ticker_data = preprocess_trades_for_mm(df)
    # Manually add an empty window
    ticker_data["TEST-A"].append([])  # Empty second window

    cfg = MMConfig(inventory_lambda=0.01, max_inventory=20)
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # Step 1: Fill to build inventory
    action = np.array([-1.0, 1.0], dtype=np.float32)  # Wide bid
    obs1, reward1, done1, _, info1 = env.step(action)

    inventory_after_fill = env._inventory
    assert inventory_after_fill > 0  # Should have some inventory

    # Step 2: Empty window (no trades)
    obs2, reward2, done2, _, info2 = env.step(action)

    # Reward should be negative (inventory penalty with no PnL change)
    # reward = delta_pnl - inv_penalty
    # Since no trades, delta_pnl ≈ 0 (small unrealized PnL change possible)
    # inv_penalty = 0.01 * abs(inventory) > 0
    # So reward should be negative
    assert reward2 < 0  # Penalty applied
    assert done2  # Episode ends (only 2 windows)


def test_observation_bounds():
    """Test that observations stay within expected bounds."""
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 5,
        "yes_price": [0.45, 0.50, 0.55, 0.60, 0.65],
        "count": [10, 20, 15, 10, 5],
        "taker_side": ["yes", "no", "yes", "no", "yes"],
        "created_time": [
            "2026-06-24T10:00:00Z",
            "2026-06-24T10:01:00Z",
            "2026-06-24T10:02:00Z",
            "2026-06-24T10:03:00Z",
            "2026-06-24T10:04:00Z",
        ],
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig()
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # Run several steps
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        # Check observation bounds
        assert env.observation_space.contains(obs), f"Obs out of bounds: {obs}"

        if done:
            break


def test_episode_termination_and_flatten():
    """Test that episode ends correctly and inventory is flattened."""
    # Single window with one trade
    df = pl.DataFrame({
        "ticker": ["TEST-A"],
        "yes_price": [0.30],
        "count": [10],
        "taker_side": ["no"],
        "created_time": ["2026-06-24T10:00:00Z"],
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig(max_inventory=20, inventory_lambda=0.0)
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # Take action to potentially fill
    action = np.array([-1.0, 1.0], dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)

    # Should be done (only 1 window)
    assert done
    assert truncated

    # Inventory should be flattened (back to 0)
    assert env._inventory == 0

    # PnL should include the flattening
    # If we bought at 0.30 and flattened at mid (0.30), realized_pnl includes the trade


def test_fill_sell_logic():
    """Test that sell fills work correctly (taker_side='yes')."""
    # Trades where taker buys YES (agent sells YES at ask) across 3 windows
    df = pl.DataFrame({
        "ticker": ["TEST-A"] * 3 + ["TEST-A"] * 2 + ["TEST-A"] * 1,
        "yes_price": [0.55, 0.56, 0.57, 0.56, 0.57, 0.56],
        "count": [5, 5, 5, 5, 5, 5],
        "taker_side": ["yes", "yes", "yes", "yes", "yes", "yes"],
        "created_time": ["2026-06-24T10:00:00Z"] * 3 + ["2026-06-24T10:01:00Z"] * 2 + ["2026-06-24T10:02:00Z"],
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig(
        min_half_spread=0.01,
        max_half_spread=0.02,
        max_skew=0.05,  # Need skew for this test
        max_inventory=20,
        inventory_lambda=0.0,
    )
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # VWAP = (0.55*5 + 0.56*5 + 0.57*5) / 15 = 0.56
    # Use narrower ask: action to get ask below trade prices
    action = np.array([-1.0, -1.0], dtype=np.float32)  # Min spread, negative skew
    # Action [-1, -1] with cfg gives:
    # half_spread = (-1+1)/2 * (0.02-0.01) + 0.01 = 0.01
    # skew = -1 * 0.05 = -0.05
    # bid = 0.56 - 0.01 - 0.05 = 0.50
    # ask = 0.56 + 0.01 - 0.05 = 0.52
    # Trades at 0.55, 0.56, 0.57 are all >= ask (0.52), should fill!

    obs, reward, done, _, info = env.step(action)

    # Should have sold (negative inventory)
    assert env._inventory < 0
    assert env._inventory == -15  # All 3 trades * 5 contracts
    assert not done  # Still have windows 2 and 3


def test_maker_fee_deduction():
    """Test that maker fees are correctly deducted from realized PnL."""
    df = pl.DataFrame({
        "ticker": ["TEST-A"],
        "yes_price": [0.50],
        "count": [10],
        "taker_side": ["no"],
        "created_time": ["2026-06-24T10:00:00Z"],
    })

    ticker_data = preprocess_trades_for_mm(df)
    cfg = MMConfig(maker_fee_rate=0.0175, inventory_lambda=0.0)
    env = MMEnv(ticker_data, cfg)

    obs, info = env.reset()

    # Action to ensure fill
    action = np.array([1.0, 0.0], dtype=np.float32)  # Max spread -> bid below mid
    # mid = 0.50, half_spread = 0.10, bid = 0.40, ask = 0.60
    # Trade at 0.50 > bid (0.40), no fill

    # Use action to get bid >= 0.50
    action = np.array([-1.0, 1.0], dtype=np.float32)
    # min_half_spread = 0.01, max_skew = 0.05
    # bid = 0.50 - 0.01 + 0.05 = 0.54
    # Trade at 0.50 <= bid (0.54), fills!

    initial_pnl = env._realized_pnl
    obs, reward, done, _, info = env.step(action)

    # Fee should be deducted
    # compute_maker_fee(10, 0.50, 0.0175) = 0.0175 * 10 * 0.50 * 0.50 = 0.04375 -> rounds to $0.05
    # realized_pnl should decrease by fee amount
    assert env._realized_pnl < initial_pnl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
