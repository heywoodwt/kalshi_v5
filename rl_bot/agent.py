import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_bot.config import RLConfig
from rl_bot.exploration import ExplorationStrategy

log = logging.getLogger(__name__)
from rl_bot import normalizer as _normalizer


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
        self, batch_size: int, rng: np.random.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch and return as tensors.

        If a numpy Generator is provided, use it for deterministic sampling;
        otherwise fall back to the stdlib random.sample.
        """
        buf_list = list(self._buffer)
        if rng is None:
            batch = random.sample(buf_list, batch_size)
        else:
            idx = rng.choice(len(buf_list), size=batch_size, replace=False)
            batch = [buf_list[i] for i in idx]

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

    def __init__(
        self,
        cfg: RLConfig,
        exploration_strategy: ExplorationStrategy | None = None,
    ) -> None:
        self._cfg = cfg
        self._strategy = exploration_strategy
        self.step_count = 0

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            log.info("Using CUDA GPU for training (%d device(s))",
                     torch.cuda.device_count())
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
            log.info("Using MPS (Apple Silicon GPU)")
        else:
            self._device = torch.device("cpu")
            log.info("Using CPU (no GPU available)")

        # Online Q-network (the one we train)
        self._q_net = DuelingDQN(
            cfg.state_dim, cfg.n_actions, cfg.hidden_dim, cfg.dueling_dim
        ).to(self._device)
        # Target Q-network (frozen copy for stable TD targets)
        self._target_net = DuelingDQN(
            cfg.state_dim, cfg.n_actions, cfg.hidden_dim, cfg.dueling_dim
        ).to(self._device)
        # Initialize target with same weights
        self._target_net.load_state_dict(self._q_net.state_dict())
        self._target_net.eval()

        # Wrap with DataParallel for multi-GPU training
        self._multi_gpu = False
        if torch.cuda.device_count() > 1:
            self._q_net = nn.DataParallel(self._q_net)
            self._target_net = nn.DataParallel(self._target_net)
            self._multi_gpu = True
            # Scale batch size by GPU count for better utilization
            self._effective_batch = cfg.batch_size * torch.cuda.device_count()
            log.info("DataParallel enabled: %d GPUs, batch %d -> %d",
                     torch.cuda.device_count(), cfg.batch_size,
                     self._effective_batch)
        else:
            self._effective_batch = cfg.batch_size

        self._optimizer = optim.Adam(self._q_net.parameters(), lr=cfg.lr)
        self._buffer = ReplayBuffer(cfg.replay_capacity)
        # RNG for agent-level stochasticity (action exploration, buffer sampling)
        seed = cfg.seed if hasattr(cfg, "seed") else None
        self._rng = np.random.default_rng(seed)

    def epsilon(self) -> float:
        """Current exploration rate, linearly decayed from eps_start to eps_end."""
        # Delegate to strategy if available
        if self._strategy is not None:
            return self._strategy.epsilon(self.step_count)

        # Backward compatible: legacy linear decay from config
        cfg = self._cfg
        if self.step_count >= cfg.eps_decay_steps:
            return cfg.eps_end
        # Linear interpolation
        frac = self.step_count / cfg.eps_decay_steps
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    def select_action(
        self, state: np.ndarray, valid_mask: np.ndarray,
        invert: bool = False,
    ) -> int:
        """Choose an action using epsilon-greedy with action masking.

        Args:
            state: feature vector of shape (state_dim,)
            valid_mask: binary mask of shape (n_actions,), 1 = valid, 0 = invalid
            invert: if True, pick the action with the LOWEST Q-value (policy
                inversion test — if the model learned real but backwards signal,
                inverting should produce positive PnL)

        Returns:
            Integer action ID (0 to n_actions-1)
        """
        # Delegate to strategy if available
        if self._strategy is not None:
            # Compute Q-values for strategy
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
                q_values = self._q_net(state_t).squeeze(0).cpu().numpy()

            return self._strategy.select_action(
                step=self.step_count,
                q_values=q_values,
                valid_mask=valid_mask,
                rng=self._rng,
            )

        # Backward compatible: legacy epsilon-greedy
        # Epsilon-greedy: explore with probability epsilon
        if self._rng.random() < self.epsilon():
            # Random valid action (use Generator for reproducibility)
            valid_actions = np.where(valid_mask > 0)[0]
            return int(self._rng.choice(valid_actions))

        # Greedy: pick action with highest (or lowest if inverted) Q-value
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self._device).unsqueeze(0)
            q_values = self._q_net(state_t).squeeze(0)  # (n_actions,)

        # Mask invalid actions
        mask_t = torch.tensor(valid_mask, dtype=torch.float32, device=self._device)
        if invert:
            # Set invalid actions to +inf so argmin ignores them
            q_values = q_values + (1.0 - mask_t) * 1e9
            return int(q_values.argmin().item())
        else:
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
        bs = self._effective_batch
        # Don't train during warmup or if buffer is too small
        if len(self._buffer) < cfg.warmup_steps or len(self._buffer) < bs:
            return None

        # Sample batch (use agent's RNG for deterministic sampling)
        states, actions, rewards, next_states, dones = self._buffer.sample(bs, rng=self._rng)

        # Move tensors to device
        states = states.to(self._device)
        actions = actions.to(self._device)
        rewards = rewards.to(self._device)
        next_states = next_states.to(self._device)
        dones = dones.to(self._device)

        # Current Q-values for chosen actions
        q_all = self._q_net(states)                    # (batch, n_actions)
        # Use squeeze(-1) to avoid removing batch dimension when batch_size==1
        q_values = q_all.gather(1, actions.unsqueeze(1)).squeeze(-1)  # (batch,)

        # Target Q-values using Double DQN (reduces overestimation):
        # 1) use online network to select next actions
        # 2) use target network to evaluate those actions
        with torch.no_grad():
            next_q_online = self._q_net(next_states)            # (batch, n_actions)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (batch, 1)
            next_q_target = self._target_net(next_states)        # (batch, n_actions)
            max_next_q = next_q_target.gather(1, next_actions).squeeze(-1)  # (batch,)
            # Bellman target: r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
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
        # Capture numpy Generator state (bit_generator.state is a plain dict)
        try:
            rng_state = self._rng.bit_generator.state
        except Exception:
            rng_state = None

        # Capture global normalizer state (means, vars, counts) for reproducibility
        try:
            obs_norm_state = {
                "mean": _normalizer.OBS_NORMALIZER.mean.copy(),
                "var": _normalizer.OBS_NORMALIZER.var.copy(),
                "count": float(_normalizer.OBS_NORMALIZER.count),
            }
            reward_norm_state = {
                "mean": float(_normalizer.REWARD_NORMALIZER.mean),
                "var": float(_normalizer.REWARD_NORMALIZER.var),
                "count": float(_normalizer.REWARD_NORMALIZER.count),
            }
        except Exception:
            obs_norm_state = None
            reward_norm_state = None

        # Python stdlib and torch RNG states
        try:
            import random as _py_random

            py_random_state = _py_random.getstate()
        except Exception:
            py_random_state = None

        try:
            torch_rng_state = torch.get_rng_state()
        except Exception:
            torch_rng_state = None

        try:
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        except Exception:
            cuda_rng_state = None

        # Unwrap DataParallel for portable checkpoints
        q_state = (self._q_net.module.state_dict() if self._multi_gpu
                   else self._q_net.state_dict())
        t_state = (self._target_net.module.state_dict() if self._multi_gpu
                   else self._target_net.state_dict())
        torch.save({
            "q_net": q_state,
            "target_net": t_state,
            "optimizer": self._optimizer.state_dict(),
            "step_count": self.step_count,
            "eps_start": self._cfg.eps_start,
            "rng_state": rng_state,
            "py_random_state": py_random_state,
            "torch_rng_state": torch_rng_state,
            "cuda_rng_state": cuda_rng_state,
            "obs_normalizer": obs_norm_state,
            "reward_normalizer": reward_norm_state,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from disk."""
        ckpt = torch.load(path, weights_only=False, map_location=self._device)
        # Load into unwrapped module if DataParallel is active
        q_mod = self._q_net.module if self._multi_gpu else self._q_net
        t_mod = self._target_net.module if self._multi_gpu else self._target_net
        q_mod.load_state_dict(ckpt["q_net"])
        t_mod.load_state_dict(ckpt["target_net"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
        # Restore numpy RNG state if present
        try:
            rng_state = ckpt.get("rng_state", None)
            if rng_state is not None:
                # recreate Generator and set internal state
                self._rng = np.random.default_rng()
                self._rng.bit_generator.state = rng_state
        except Exception:
            pass

        # Restore python random state
        try:
            py_state = ckpt.get("py_random_state", None)
            if py_state is not None:
                import random as _py_random

                _py_random.setstate(py_state)
        except Exception:
            pass

        # Restore torch RNG state
        try:
            tr_state = ckpt.get("torch_rng_state", None)
            if tr_state is not None:
                torch.set_rng_state(tr_state)
            cuda_state = ckpt.get("cuda_rng_state", None)
            if cuda_state is not None and torch.cuda.is_available():
                # cuda_state is a list of tensors
                try:
                    torch.cuda.set_rng_state_all(cuda_state)
                except Exception:
                    # older torch versions may not have set_rng_state_all
                    for i, st in enumerate(cuda_state):
                        try:
                            torch.cuda.set_rng_state(st, device=i)
                        except Exception:
                            pass
        except Exception:
            pass

        # Restore normalizers
        try:
            obs_state = ckpt.get("obs_normalizer", None)
            if obs_state is not None:
                _normalizer.OBS_NORMALIZER.mean = np.array(obs_state["mean"], dtype=np.float64)
                _normalizer.OBS_NORMALIZER.var = np.array(obs_state["var"], dtype=np.float64)
                _normalizer.OBS_NORMALIZER.count = float(obs_state["count"])

            r_state = ckpt.get("reward_normalizer", None)
            if r_state is not None:
                _normalizer.REWARD_NORMALIZER.mean = float(r_state["mean"])
                _normalizer.REWARD_NORMALIZER.var = float(r_state["var"])
                _normalizer.REWARD_NORMALIZER.count = float(r_state["count"])
        except Exception:
            pass
