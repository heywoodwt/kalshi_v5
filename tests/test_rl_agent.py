import numpy as np
import pytest
import torch


def test_dueling_dqn_output_shape():
    from rl_bot.agent import DuelingDQN
    net = DuelingDQN(state_dim=18, n_actions=21, hidden_dim=128, dueling_dim=64)
    x = torch.randn(4, 18)
    q = net(x)
    assert q.shape == (4, 21)


def test_dueling_dqn_single_sample():
    from rl_bot.agent import DuelingDQN
    net = DuelingDQN(state_dim=18, n_actions=21, hidden_dim=128, dueling_dim=64)
    x = torch.randn(1, 18)
    q = net(x)
    assert q.shape == (1, 21)
    # Q-values should be finite
    assert torch.isfinite(q).all()


def test_replay_buffer_push_and_len():
    from rl_bot.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0
    state = np.zeros(18, dtype=np.float32)
    buf.push(state, 0, 1.0, state, False)
    assert len(buf) == 1


def test_replay_buffer_sample():
    from rl_bot.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=100)
    state = np.zeros(18, dtype=np.float32)
    for i in range(10):
        buf.push(state, i % 21, float(i), state, False)
    states, actions, rewards, next_states, dones = buf.sample(4)
    assert states.shape == (4, 18)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 18)
    assert dones.shape == (4,)


def test_replay_buffer_overflow():
    from rl_bot.agent import ReplayBuffer
    buf = ReplayBuffer(capacity=5)
    state = np.zeros(18, dtype=np.float32)
    for i in range(10):
        buf.push(state, 0, float(i), state, False)
    # Should not exceed capacity
    assert len(buf) == 5


def test_agent_select_action_shape():
    from rl_bot.config import RLConfig
    from rl_bot.agent import DQNAgent
    cfg = RLConfig(eps_start=0.0, eps_end=0.0)  # greedy, no randomness
    agent = DQNAgent(cfg)
    state = np.zeros(18, dtype=np.float32)
    mask = np.ones(21, dtype=np.float32)  # all actions valid
    action = agent.select_action(state, mask)
    assert 0 <= action <= 20


def test_agent_action_masking():
    from rl_bot.config import RLConfig
    from rl_bot.agent import DQNAgent
    cfg = RLConfig(eps_start=0.0, eps_end=0.0)  # pure greedy
    agent = DQNAgent(cfg)
    state = np.zeros(18, dtype=np.float32)
    # Only HOLD (action 18) is valid
    mask = np.zeros(21, dtype=np.float32)
    mask[18] = 1.0
    action = agent.select_action(state, mask)
    assert action == 18


def test_agent_epsilon_decay():
    from rl_bot.config import RLConfig
    from rl_bot.agent import DQNAgent
    cfg = RLConfig(eps_start=1.0, eps_end=0.1, eps_decay_steps=100)
    agent = DQNAgent(cfg)
    assert agent.epsilon() == 1.0
    # Simulate 50 steps
    agent.step_count = 50
    eps = agent.epsilon()
    assert 0.1 < eps < 1.0
    # At 100+ steps, should be at minimum
    agent.step_count = 100
    assert agent.epsilon() == 0.1


def test_agent_train_step_returns_none_during_warmup():
    from rl_bot.config import RLConfig
    from rl_bot.agent import DQNAgent
    cfg = RLConfig(warmup_steps=100, batch_size=4)
    agent = DQNAgent(cfg)
    # No data in buffer
    result = agent.train_step()
    assert result is None


def test_agent_train_step_runs_after_warmup():
    from rl_bot.config import RLConfig
    from rl_bot.agent import DQNAgent
    cfg = RLConfig(warmup_steps=5, batch_size=4)
    agent = DQNAgent(cfg)
    state = np.random.randn(18).astype(np.float32)
    # Fill buffer past warmup
    for i in range(10):
        agent.store_transition(state, i % 21, 0.1, state, False)
    agent.step_count = 10
    loss = agent.train_step()
    assert loss is not None
    assert loss >= 0.0


def test_agent_checkpoint_roundtrip(tmp_path):
    from rl_bot.config import RLConfig
    from rl_bot.agent import DQNAgent
    cfg = RLConfig()
    agent = DQNAgent(cfg)
    agent.step_count = 42
    path = str(tmp_path / "test_ckpt.pt")
    agent.save_checkpoint(path)

    agent2 = DQNAgent(cfg)
    agent2.load_checkpoint(path)
    assert agent2.step_count == 42
