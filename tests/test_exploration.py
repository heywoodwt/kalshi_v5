import pytest
import numpy as np
from rl_bot.exploration import ExplorationStrategy, FastLinearDecay, ExponentialDecay, LogarithmicDecay, EpisodeBased


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


def test_exponential_decay_epsilon():
    """Verify exponential decay formula and floor."""
    strategy = ExponentialDecay({
        "eps_start": 0.8,
        "eps_end": 0.05,
        "decay_rate": 0.99,
    })

    assert strategy.epsilon(0) == 0.8
    # After 100 steps: 0.8 * 0.99^100 ≈ 0.2907
    assert strategy.epsilon(100) == pytest.approx(max(0.05, 0.8 * 0.99**100), abs=1e-4)
    # After 1000 steps: hits floor
    assert strategy.epsilon(1000) == 0.05


def test_exponential_decay_action_selection():
    """Verify action selection is epsilon-greedy."""
    strategy = ExponentialDecay({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_rate": 0.99,
    })

    rng = np.random.default_rng(123)
    q_values = np.array([5.0, 1.0, 2.0])
    valid_mask = np.ones(3)

    # High epsilon: mostly random
    actions = [strategy.select_action(0, q_values, valid_mask, rng) for _ in range(50)]
    assert len(set(actions)) >= 2  # some exploration


def test_logarithmic_decay_epsilon():
    """Verify logarithmic decay shape (fast early, slow late)."""
    strategy = LogarithmicDecay({
        "eps_start": 0.8,
        "eps_end": 0.05,
        "decay_steps": 1000,
    })

    eps_0 = strategy.epsilon(0)
    eps_100 = strategy.epsilon(100)
    eps_500 = strategy.epsilon(500)
    eps_1000 = strategy.epsilon(1000)

    # Monotonic decreasing
    assert eps_0 > eps_100 > eps_500 > eps_1000
    # Floor at decay_steps
    assert eps_1000 == 0.05
    # Start value
    assert eps_0 == 0.8


def test_logarithmic_decay_action_selection():
    """Verify epsilon-greedy behavior."""
    strategy = LogarithmicDecay({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_steps": 100,
    })

    rng = np.random.default_rng(99)
    q_values = np.array([10.0, 1.0])
    valid_mask = np.ones(2)

    # At floor (step 100), mostly greedy
    actions = [strategy.select_action(100, q_values, valid_mask, rng) for _ in range(30)]
    assert actions.count(0) >= 25  # mostly action 0 (greedy)


def test_episode_based_decay():
    """Verify episode-based exponential decay."""
    strategy = EpisodeBased({
        "eps_start": 0.8,
        "eps_end": 0.05,
        "decay_rate": 0.99,
    })

    # Episode 0
    strategy.episode_count = 0
    assert strategy.epsilon(step=0) == 0.8

    # Episode 100
    strategy.episode_count = 100
    expected = max(0.05, 0.8 * 0.99**100)
    assert strategy.epsilon(step=999) == pytest.approx(expected, abs=1e-4)

    # Episode 500 (hits floor)
    strategy.episode_count = 500
    assert strategy.epsilon(step=9999) == 0.05


def test_episode_based_action_selection():
    """Verify epsilon-greedy behavior."""
    strategy = EpisodeBased({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_rate": 0.99,
    })
    strategy.episode_count = 0  # high epsilon

    rng = np.random.default_rng(777)
    q_values = np.array([5.0, 1.0, 2.0])
    valid_mask = np.ones(3)

    # Should explore
    actions = [strategy.select_action(0, q_values, valid_mask, rng) for _ in range(30)]
    assert len(set(actions)) >= 2


def test_action_local_explores_same_direction():
    """Verify action-local explores within same direction for buy actions."""
    from rl_bot.exploration import ActionLocal

    strategy = ActionLocal({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_steps": 100,
    })

    rng = np.random.default_rng(42)

    # Mock Q-values where greedy action is BUY_YES_1_AT_0c (action 0)
    # Action space: 0-8=YES, 9-17=NO, 18=HOLD, 19=CLOSE_YES, 20=CLOSE_NO
    q_values = np.zeros(21)
    q_values[0] = 10.0  # highest Q
    valid_mask = np.ones(21)

    # With eps=1.0 (step 0), should explore within YES direction (0-8) or HOLD
    actions = [strategy.select_action(0, q_values, valid_mask, rng) for _ in range(100)]

    # No NO trades (9-17) when greedy action is YES
    assert all(a not in range(9, 18) for a in actions)

    # Should explore different YES actions (not all action 0)
    yes_actions = [a for a in actions if 0 <= a <= 8]
    assert len(set(yes_actions)) > 1


def test_action_local_greedy_at_low_epsilon():
    """At low epsilon, should be mostly greedy."""
    from rl_bot.exploration import ActionLocal

    strategy = ActionLocal({
        "eps_start": 1.0,
        "eps_end": 0.05,
        "decay_steps": 100,
    })

    rng = np.random.default_rng(555)
    q_values = np.zeros(21)
    q_values[5] = 20.0  # greedy action
    valid_mask = np.ones(21)

    # At step 100, eps=0.05
    actions = [strategy.select_action(100, q_values, valid_mask, rng) for _ in range(40)]
    # Most should be greedy (action 5)
    assert actions.count(5) >= 35
