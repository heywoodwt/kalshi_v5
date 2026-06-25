import pytest
import numpy as np


def test_mm_config_defaults():
    """Test all default values of MMConfig fields."""
    from rl_bot.mm_config import MMConfig
    cfg = MMConfig()

    # Observation
    assert cfg.obs_dim == 10

    # Action bounds
    assert cfg.min_half_spread == 0.01
    assert cfg.max_half_spread == 0.10
    assert cfg.max_skew == 0.05

    # Inventory & risk
    assert cfg.max_inventory == 20
    assert cfg.inventory_lambda == 0.01
    assert cfg.inventory_tte_scale is True

    # Fees
    assert cfg.maker_fee_rate == 0.0175
    assert cfg.taker_fee_rate == 0.07

    # Quoting
    assert cfg.quote_size == 1

    # PPO training
    assert cfg.total_timesteps == 500_000
    assert cfg.n_epochs_ppo == 10
    assert cfg.learning_rate == 3e-4
    assert cfg.gamma == 0.99
    assert cfg.batch_size == 64
    assert cfg.n_steps == 2048

    # Data
    assert cfg.min_trades_per_ticker == 50


def test_scale_action_center():
    """Test scale_action at center point (0, 0)."""
    from rl_bot.mm_config import MMConfig, scale_action
    cfg = MMConfig()
    raw = np.array([0.0, 0.0])
    half_spread, skew = scale_action(raw, cfg)

    # Center of [0.01, 0.10] is 0.055
    assert half_spread == pytest.approx(0.055)
    # Center of [-0.05, 0.05] is 0.0
    assert skew == pytest.approx(0.0)


def test_scale_action_extremes_neg():
    """Test scale_action at negative extreme (-1, -1)."""
    from rl_bot.mm_config import MMConfig, scale_action
    cfg = MMConfig()
    raw = np.array([-1.0, -1.0])
    half_spread, skew = scale_action(raw, cfg)

    # -1 maps to min_half_spread
    assert half_spread == pytest.approx(0.01)
    # -1 maps to -max_skew
    assert skew == pytest.approx(-0.05)


def test_scale_action_extremes_pos():
    """Test scale_action at positive extreme (1, 1)."""
    from rl_bot.mm_config import MMConfig, scale_action
    cfg = MMConfig()
    raw = np.array([1.0, 1.0])
    half_spread, skew = scale_action(raw, cfg)

    # 1 maps to max_half_spread
    assert half_spread == pytest.approx(0.10)
    # 1 maps to max_skew
    assert skew == pytest.approx(0.05)


def test_scale_action_edge_values():
    """Test scale_action with various edge values."""
    from rl_bot.mm_config import MMConfig, scale_action
    cfg = MMConfig()

    # Test (1, -1): max spread, negative skew
    raw = np.array([1.0, -1.0])
    half_spread, skew = scale_action(raw, cfg)
    assert half_spread == pytest.approx(0.10)
    assert skew == pytest.approx(-0.05)

    # Test (-1, 1): min spread, positive skew
    raw = np.array([-1.0, 1.0])
    half_spread, skew = scale_action(raw, cfg)
    assert half_spread == pytest.approx(0.01)
    assert skew == pytest.approx(0.05)

    # Test (0.5, 0.0): 3/4 of spread range, no skew
    raw = np.array([0.5, 0.0])
    half_spread, skew = scale_action(raw, cfg)
    # 0.5 → (0.5+1)/2 = 0.75 → 0.75 * 0.09 + 0.01 = 0.0775
    assert half_spread == pytest.approx(0.0775)
    assert skew == pytest.approx(0.0)

    # Test (0.0, 0.5): center spread, half positive skew
    raw = np.array([0.0, 0.5])
    half_spread, skew = scale_action(raw, cfg)
    assert half_spread == pytest.approx(0.055)
    # 0.5 * 0.05 = 0.025
    assert skew == pytest.approx(0.025)


def test_scale_action_output_types():
    """Test that scale_action returns proper float types."""
    from rl_bot.mm_config import MMConfig, scale_action
    cfg = MMConfig()
    raw = np.array([0.0, 0.0])
    half_spread, skew = scale_action(raw, cfg)

    # Ensure outputs are Python floats (not numpy types)
    assert isinstance(half_spread, float)
    assert isinstance(skew, float)
