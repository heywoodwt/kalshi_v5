from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    # Network architecture
    state_dim: int = 18
    n_actions: int = 21
    hidden_dim: int = 128
    dueling_dim: int = 64

    # Replay buffer
    replay_capacity: int = 100_000
    batch_size: int = 64

    # Training hyperparameters
    gamma: float = 0.99
    lr: float = 1e-4
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    target_update_freq: int = 1_000
    warmup_steps: int = 1_000

    # Decision cadence
    decision_interval_s: float = 30.0

    # Risk controls
    max_position_per_market: int = 5
    max_total_markets: int = 10
    max_daily_loss: float = 50.0

    # Persistence
    checkpoint_freq: int = 1_000
    checkpoint_dir: str = "rl_bot/checkpoints"
    log_csv_path: str = "output/rl_trades.csv"

    # Trading mode
    paper_trading: bool = True

    # Reproducibility: seeds the agent's exploration RNG (see DQNAgent.__init__).
    # None leaves the RNG unseeded.
    seed: int | None = None

    # Fee schedule
    maker_fee_rate: float = 0.0175

    # Action space parameters
    sizes: tuple[int, ...] = (1, 3, 5)
    offsets: tuple[float, ...] = (0.0, 0.02, 0.04)


# Control action indices (after the 18 buy actions)
ACTION_HOLD = 18
ACTION_CLOSE_YES = 19
ACTION_CLOSE_NO = 20

# Module-level constants for decode_action (avoid re-creating RLConfig each call)
_SIZES = (1, 3, 5)
_OFFSETS = (0.0, 0.02, 0.04)


def decode_action(action_id: int) -> tuple[str, int, float] | str:
    """Decode an integer action ID into its semantic meaning.

    Buy actions (0-17): returns (direction, size, offset)
      direction * 9 + size_idx * 3 + offset_idx
    Control actions (18-20): returns "hold", "close_yes", or "close_no"

    Args:
        action_id: integer action identifier (0-20)

    Returns:
        For buy actions: tuple of (direction: str, size: int, offset: float)
        For control actions: string ("hold", "close_yes", or "close_no")

    Raises:
        ValueError: if action_id is outside the range [0, 20]
    """
    # Validate action_id is within valid range
    if action_id < 0 or action_id > 20:
        raise ValueError(f"Invalid action_id: {action_id}")

    # Handle control actions
    if action_id == ACTION_HOLD:
        return "hold"
    if action_id == ACTION_CLOSE_YES:
        return "close_yes"
    if action_id == ACTION_CLOSE_NO:
        return "close_no"

    # Decode buy actions (0-17)
    # Action space: 2 directions × 3 sizes × 3 offsets = 18 actions
    # Encoding: direction_idx * 9 + size_idx * 3 + offset_idx
    direction_idx = action_id // 9       # 0 = yes, 1 = no (9 = 3 sizes * 3 offsets)
    remainder = action_id % 9
    size_idx = remainder // 3            # 0, 1, 2
    offset_idx = remainder % 3           # 0, 1, 2

    direction = "yes" if direction_idx == 0 else "no"
    size = _SIZES[size_idx]
    offset = _OFFSETS[offset_idx]

    return (direction, size, offset)
