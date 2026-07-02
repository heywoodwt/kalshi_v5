"""Tests for the sim-vs-live adverse-selection bias fixes (docs/sim_vs_live_bias.md).

Covers:
- taker fee helper and new MMConfig knobs
- post-fill inventory marking in MMEnv.step() reward
- taker-fee fraction charged on sim fills
- 20th observation dimension (realized volatility), sim and live formulas
"""
import numpy as np
import polars as pl
import pytest

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.reward import compute_maker_fee, compute_taker_fee


def test_taker_fee_uses_taker_rate():
    # Same variance formula as maker fee, higher rate: 0.07 * 10 * 0.5 * 0.5 = 0.175 -> ceil to 0.18
    assert compute_taker_fee(10, 0.50, 0.07) == 0.18
    # Taker fee must exceed maker fee at the same price/size
    assert compute_taker_fee(10, 0.50, 0.07) > compute_maker_fee(10, 0.50, 0.0175)


def test_config_bias_fix_defaults():
    cfg = MMConfig()
    assert cfg.taker_fee_rate == 0.07
    assert cfg.taker_fill_prob == 0.10  # post-only quoting: only exits cross
    assert cfg.vol_filter_threshold == 0.05
    assert cfg.through_fill_haircut == 0.33


# --- Post-fill marking -----------------------------------------------------------


def _two_window_ticker(second_vwap: float) -> pl.DataFrame:
    """Two 60s windows for one ticker.

    Window 1 (12:00): a big print at 0.50 anchors the VWAP high (~0.47), and a
    10-lot at 0.20 (taker_side="no" -> we buy YES) crosses any bid near 0.22.
    Window 2 (12:01): a single print sets the post-fill mark.
    """
    return pl.DataFrame({
        "ticker": ["TEST"] * 3,
        "yes_price": [0.50, 0.20, second_vwap],
        "count": [100, 10, 1],
        "taker_side": ["yes", "no", "yes"],
        "created_time": ["2026-06-25T12:00:00Z", "2026-06-25T12:00:30Z", "2026-06-25T12:01:00Z"],
    })


def test_reward_marks_inventory_at_post_fill_mid():
    ticker_data = preprocess_mm_data(_two_window_ticker(second_vwap=0.10))
    cfg = MMConfig(taker_fill_prob=0.0)  # isolate the marking change from the fee change
    env = MMEnv(ticker_data=ticker_data, cfg=cfg)
    env.reset(seed=0)

    # action [0,0] -> half_spread 0.255 -> bid ~0.22, so the 0.20 print fills us long
    obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

    assert info["fills_buy"] > 0  # sanity: the fill happened
    # We bought at 0.20 and the next window marks at 0.10: reward must be negative.
    # (The old biased code marked at the same window's VWAP ~0.47 -> large positive reward.)
    assert reward < 0


# --- Taker-fee fraction on fills ---------------------------------------------------


def _make_env(taker_fill_prob: float) -> MMEnv:
    ticker_data = preprocess_mm_data(_two_window_ticker(second_vwap=0.50))
    cfg = MMConfig(taker_fill_prob=taker_fill_prob)
    env = MMEnv(ticker_data=ticker_data, cfg=cfg)
    env.reset(seed=0)
    return env


def test_fills_charge_taker_fee_fraction():
    # prob=1.0 -> every fill pays taker rate; prob=0.0 -> maker rate.
    # Fees are at quote_size (=1) granularity: per-contract ceil'd fee.
    from rl_bot.reward import fee_at_quote_size
    env_taker = _make_env(taker_fill_prob=1.0)
    env_taker._fill_buy(0.50, 10)
    assert env_taker._realized_pnl == -fee_at_quote_size(10, 0.50, 0.07, 1)  # -0.20

    env_maker = _make_env(taker_fill_prob=0.0)
    env_maker._fill_buy(0.50, 10)
    assert env_maker._realized_pnl == -fee_at_quote_size(10, 0.50, 0.0175, 1)  # -0.10


# --- Realized volatility observation (dim 20) --------------------------------------


def _four_window_ticker() -> pl.DataFrame:
    """Four windows with swinging VWAPs so mid history builds up variance."""
    return pl.DataFrame({
        "ticker": ["TEST"] * 4,
        "yes_price": [0.50, 0.30, 0.60, 0.20],
        "count": [1, 1, 1, 1],
        "taker_side": ["yes", "yes", "yes", "yes"],
        "created_time": [
            "2026-06-25T12:00:00Z",
            "2026-06-25T12:01:00Z",
            "2026-06-25T12:02:00Z",
            "2026-06-25T12:03:00Z",
        ],
    })


def test_observation_is_20_dim_with_volatility():
    ticker_data = preprocess_mm_data(_two_window_ticker(second_vwap=0.10))
    env = MMEnv(ticker_data=ticker_data, cfg=MMConfig())
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == (20,)
    assert obs.shape == (20,)
    assert obs[19] == 0.0  # no mid history yet


def test_volatility_feature_rises_with_price_swings():
    ticker_data = preprocess_mm_data(_four_window_ticker())
    env = MMEnv(ticker_data=ticker_data, cfg=MMConfig())
    env.reset(seed=0)
    # Widest spread action -> no fills; mids 0.50, 0.30, 0.60 accumulate in history
    action = np.array([1.0, 0.0], dtype=np.float32)
    env.step(action)
    env.step(action)
    obs, *_ = env.step(action)
    # std([0.50, 0.30, 0.60]) ~ 0.125 -> normalized by 0.05 saturates at 1.0
    assert obs[19] == pytest.approx(1.0)


def test_live_realized_vol_matches_sim_formula():
    from rl_bot.live_trader_v2 import realized_vol
    assert realized_vol([0.5, 0.5]) == 0.0  # <3 samples: not enough signal
    hist = [0.40, 0.50, 0.60]
    assert realized_vol(hist) == pytest.approx(float(np.std(hist)))


# --- Phase 3: remaining sim-optimism fixes ------------------------------------------


def test_through_fill_executes_at_our_quote_not_print_price():
    # Window 1 has a 0.20 print crossing our ~0.218 bid. A real resting order
    # fills at ITS OWN price (price-time priority), not at the print price —
    # filling at 0.20 credited phantom edge on every through-fill.
    ticker_data = preprocess_mm_data(_two_window_ticker(second_vwap=0.50))
    cfg = MMConfig(taker_fill_prob=0.0)
    env = MMEnv(ticker_data=ticker_data, cfg=cfg)
    env.reset(seed=0)
    obs, reward, done, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert info["fills_buy"] > 0
    # Entry price must equal our bid (0.218), not the 0.20 print
    assert env._avg_entry_price == pytest.approx(info["bid"])
    assert env._avg_entry_price > 0.20


def test_fee_at_quote_size_matches_live_order_granularity():
    from rl_bot.reward import fee_at_quote_size
    # Live quotes are 1-lot orders: a 10-contract sim fill is really 10 orders,
    # each paying the ceil'd 1-lot fee (1c at 0.50) -> $0.10, not ceil(0.04375)=$0.05
    assert fee_at_quote_size(10, 0.50, 0.0175, quote_size=1) == pytest.approx(0.10)
    # At quote_size=10 the same fill is one order -> single ceil'd fee
    assert fee_at_quote_size(10, 0.50, 0.0175, quote_size=10) == pytest.approx(0.05)
    # Partial last lot still pays a full ceil'd per-lot fee (pessimistic)
    assert fee_at_quote_size(3, 0.50, 0.0175, quote_size=2) == pytest.approx(0.02)


def test_sim_fills_charge_fees_at_quote_size():
    # MMConfig default quote_size=1 -> per-contract ceil'd fee, matching live
    env = _make_env(taker_fill_prob=0.0)
    env._fill_buy(0.50, 10)
    assert env._realized_pnl == pytest.approx(-0.10)  # 10 x 1c, not 5c


def test_episode_end_flatten_charges_taker_fee():
    # The flatten crosses the spread — that is a taker execution and must pay
    # the taker rate (7%), not the maker rate the old code charged.
    from rl_bot.reward import fee_at_quote_size
    ticker_data = preprocess_mm_data(_two_window_ticker(second_vwap=0.50))
    cfg = MMConfig(taker_fill_prob=0.0)
    env = MMEnv(ticker_data=ticker_data, cfg=cfg)
    env.reset(seed=0)
    env.step(np.array([0.0, 0.0], dtype=np.float32))   # window 1: through-fill
    inv = env._inventory
    entry = env._avg_entry_price
    entry_fees = -env._realized_pnl                     # only fees booked so far
    assert inv > 0
    obs, reward, done, *_ = env.step(np.array([1.0, 0.0], dtype=np.float32))  # window 2: quote wide, episode ends
    assert done
    # Reconstruct the flatten: mid=0.50 (window 2 VWAP), fallback spread 0.03
    exit_price = 0.50 - 0.015
    expected_exit_fee = fee_at_quote_size(inv, exit_price, cfg.taker_fee_rate, cfg.quote_size)
    expected_pnl = inv * (exit_price - entry) - entry_fees - expected_exit_fee
    assert env._realized_pnl == pytest.approx(expected_pnl)


def test_taker_fill_prob_default_reflects_post_only_quoting():
    # After Phase 1 (post-only quotes) the measured taker fraction should be
    # near zero; 0.945 was the pre-fix live pathology. Keep a small residual
    # for stop-loss/expiry exits that deliberately cross.
    assert MMConfig().taker_fill_prob == 0.10
