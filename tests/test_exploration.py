import pytest
import numpy as np
from rl_bot.exploration import ExplorationStrategy, FastLinearDecay


def test_base_class_cannot_be_instantiated():
    """ExplorationStrategy is abstract and cannot be instantiated."""
    with pytest.raises(TypeError):
        ExplorationStrategy({})


def test_fast_linear_decay_epsilon():
    """Verify linear epsilon decay reaches floor at expected step."""
    strategy = FastLinearDecay({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_steps": 100,
    })

    assert strategy.epsilon(0) == 1.0
    assert strategy.epsilon(50) == pytest.approx(0.525)
    assert strategy.epsilon(100) == 0.05
    assert strategy.epsilon(200) == 0.05  # stays at floor


def test_fast_linear_decay_action_selection():
    """Verify action selection respects epsilon and valid mask."""
    strategy = FastLinearDecay({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_steps": 100,
    })

    rng = np.random.default_rng(42)
    q_values = np.array([10.0, 2.0, 3.0])  # greedy action = 0
    valid_mask = np.array([1.0, 1.0, 0.0])  # action 2 invalid

    # At step 0, epsilon=1.0 (always explore)
    actions = [strategy.select_action(0, q_values, valid_mask, rng) for _ in range(20)]
    # Should never select invalid action 2
    assert all(a in [0, 1] for a in actions)
    # Should have some randomness (not all action 0)
    assert len(set(actions)) > 1

    # At step 100, epsilon=0.05 (mostly greedy)
    actions = [strategy.select_action(100, q_values, valid_mask, rng) for _ in range(20)]
    # Most should be greedy (action 0)
    assert actions.count(0) >= 15
