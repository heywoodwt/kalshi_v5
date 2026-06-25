from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MMConfig:
    """Configuration for market-making PPO agent.

    This config defines the observation space, action bounds, inventory/risk
    parameters, fees, and PPO training hyperparameters for a market-agnostic
    market-making system.
    """
    # ── Observation (market-agnostic) ──
    obs_dim: int = 10

    # ── Action bounds (agent outputs [-1,1]², scaled to these) ──
    min_half_spread: float = 0.01   # 1¢ min half-spread
    max_half_spread: float = 0.10   # 10¢ max half-spread
    max_skew: float = 0.05          # ±5¢ quote skew

    # ── Inventory & risk ──
    max_inventory: int = 20
    inventory_lambda: float = 0.01  # base penalty
    inventory_tte_scale: bool = True  # scale by 1/sqrt(tte_h+1)

    # ── Fees (Kalshi) ──
    maker_fee_rate: float = 0.0175
    taker_fee_rate: float = 0.07

    # ── Quoting ──
    quote_size: int = 1  # contracts per quote (v1: fixed)

    # ── PPO training ──
    total_timesteps: int = 500_000
    n_epochs_ppo: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    n_steps: int = 2048

    # ── Data ──
    min_trades_per_ticker: int = 50  # skip illiquid tickers


def scale_action(raw: np.ndarray, cfg: MMConfig) -> tuple[float, float]:
    """Map raw agent output [-1,1]² to (half_spread, skew) in cents.

    Linear interpolation:
      - raw[0] in [-1, 1] → half_spread in [min_half_spread, max_half_spread]
      - raw[1] in [-1, 1] → skew in [-max_skew, max_skew]

    Args:
        raw: 2D numpy array of shape (2,) with values in [-1, 1]
        cfg: MMConfig instance with action bounds

    Returns:
        tuple[float, float]: (half_spread, skew) both in cents (dollars)

    Example:
        >>> cfg = MMConfig()
        >>> scale_action(np.array([0.0, 0.0]), cfg)
        (0.055, 0.0)
        >>> scale_action(np.array([-1.0, -1.0]), cfg)
        (0.01, -0.05)
        >>> scale_action(np.array([1.0, 1.0]), cfg)
        (0.10, 0.05)
    """
    # Map raw[0] from [-1, 1] to [min_half_spread, max_half_spread]
    # Formula: x in [-1,1] → x' = (x+1)/2 * (max-min) + min
    half_spread = (raw[0] + 1.0) / 2.0 * (cfg.max_half_spread - cfg.min_half_spread) + cfg.min_half_spread

    # Map raw[1] from [-1, 1] to [-max_skew, max_skew]
    # Formula: y in [-1,1] → y' = y * max_skew
    skew = raw[1] * cfg.max_skew

    return (float(half_spread), float(skew))
