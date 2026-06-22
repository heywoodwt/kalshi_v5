import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_bot.config import RLConfig


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network.

    Splits the final layer into value and advantage streams.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    This helps the network learn which states are valuable independent
    of which specific action is taken.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        dueling_dim: int,
    ) -> None:
        super().__init__()
        # Shared feature extraction layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream: how good is this state overall
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, dueling_dim),
            nn.ReLU(),
            nn.Linear(dueling_dim, 1),
        )
        # Advantage stream: how much better is each action than average
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, dueling_dim),
            nn.ReLU(),
            nn.Linear(dueling_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions.

        Args:
            x: state tensor of shape (batch, state_dim)

        Returns:
            Q-values of shape (batch, n_actions)
        """
        features = self.feature(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, n_actions)
        # Combine: Q = V + A - mean(A)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


class ReplayBuffer:
    """Fixed-capacity ring buffer for experience replay.

    Stores (state, action, reward, next_state, done) transitions.
    Uses a deque for O(1) append with automatic eviction of oldest entries.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch and return as tensors."""
        batch = random.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    """Dueling DQN agent with epsilon-greedy exploration and target network.

    Manages the Q-network, target network, replay buffer, and training loop.
    Call select_action() for decisions, store_transition() to record outcomes,
    and train_step() to run one gradient update.
    """

    def __init__(self, cfg: RLConfig) -> None:
        self._cfg = cfg
        self.step_count = 0

        # Online Q-network (the one we train)
        self._q_net = DuelingDQN(
            cfg.state_dim, cfg.n_actions, cfg.hidden_dim, cfg.dueling_dim
        )
        # Target Q-network (frozen copy for stable TD targets)
        self._target_net = DuelingDQN(
            cfg.state_dim, cfg.n_actions, cfg.hidden_dim, cfg.dueling_dim
        )
        # Initialize target with same weights
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(self._q_net.parameters(), lr=cfg.lr)
        self._buffer = ReplayBuffer(cfg.replay_capacity)

    def epsilon(self) -> float:
        """Current exploration rate, linearly decayed from eps_start to eps_end."""
        cfg = self._cfg
        if self.step_count >= cfg.eps_decay_steps:
            return cfg.eps_end
        # Linear interpolation
        frac = self.step_count / cfg.eps_decay_steps
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    def select_action(self, state: np.ndarray, valid_mask: np.ndarray) -> int:
        """Choose an action using epsilon-greedy with action masking.

        Args:
            state: feature vector of shape (state_dim,)
            valid_mask: binary mask of shape (n_actions,), 1 = valid, 0 = invalid

        Returns:
            Integer action ID (0 to n_actions-1)
        """
        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self.epsilon():
            # Random valid action
            valid_actions = np.where(valid_mask > 0)[0]
            return int(np.random.choice(valid_actions))

        # Greedy: pick action with highest Q-value among valid actions
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self._q_net(state_t).squeeze(0)  # (n_actions,)

        # Mask invalid actions by setting their Q-values to -inf
        mask_t = torch.tensor(valid_mask, dtype=torch.float32)
        q_values = q_values + (mask_t - 1.0) * 1e9  # invalid -> -1e9

        return int(q_values.argmax().item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self._buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Run one gradient step on a batch from the replay buffer.

        Returns:
            Loss value, or None if not enough data for training.
        """
        cfg = self._cfg
        # Don't train during warmup or if buffer is too small
        if len(self._buffer) < cfg.warmup_steps or len(self._buffer) < cfg.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self._buffer.sample(cfg.batch_size)

        # Current Q-values for chosen actions
        q_all = self._q_net(states)                    # (batch, n_actions)
        q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Target Q-values using target network (no gradient)
        with torch.no_grad():
            next_q = self._target_net(next_states)     # (batch, n_actions)
            max_next_q = next_q.max(dim=1).values      # (batch,)
            # Bellman target: r + gamma * max_Q(s', a') * (1 - done)
            targets = rewards + cfg.gamma * max_next_q * (1.0 - dones)

        # MSE loss between current Q and target Q
        loss = nn.functional.mse_loss(q_values, targets)

        # Gradient step
        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self._q_net.parameters(), max_norm=10.0)
        self._optimizer.step()

        # Periodically update target network
        if self.step_count % cfg.target_update_freq == 0:
            self._target_net.load_state_dict(self._q_net.state_dict())

        return float(loss.item())

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to disk."""
        torch.save({
            "q_net": self._q_net.state_dict(),
            "target_net": self._target_net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "step_count": self.step_count,
            "eps_start": self._cfg.eps_start,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from disk."""
        ckpt = torch.load(path, weights_only=False)
        self._q_net.load_state_dict(ckpt["q_net"])
        self._target_net.load_state_dict(ckpt["target_net"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
