"""Exploration strategies for DQN agent.

Provides abstract base class and concrete implementations for epsilon-greedy
and alternative exploration approaches. Each strategy controls both the
exploration rate (epsilon) and the action selection logic.
"""
from abc import ABC, abstractmethod
import math
import numpy as np


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies.

    Subclasses must implement epsilon() and select_action() methods.
    The strategy is responsible for both decay schedule and action sampling.
    """

    def __init__(self, config: dict):
        """Initialize strategy with hyperparameters.

        Args:
            config: Dict of strategy-specific parameters
        """
        self.config = config

    @abstractmethod
    def epsilon(self, step: int) -> float:
        """Return exploration rate at given training step.

        Args:
            step: Current training step (agent.step_count)

        Returns:
            Epsilon value in [0, 1]
        """
        pass

    @abstractmethod
    def select_action(
        self,
        step: int,
        q_values: np.ndarray,
        valid_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Select action given Q-values and validity mask.

        Args:
            step: Current training step
            q_values: Raw Q-values from network, shape (n_actions,)
            valid_mask: Binary mask, 1=valid, 0=invalid, shape (n_actions,)
            rng: Numpy random generator for deterministic sampling

        Returns:
            Integer action ID [0, 20]
        """
        pass


class FastLinearDecay(ExplorationStrategy):
    """Linear epsilon decay from eps_start to eps_end over decay_steps.

    Action selection: standard epsilon-greedy (random valid action vs. greedy).
    """

    def epsilon(self, step: int) -> float:
        """Linear interpolation from start to end."""
        eps_start = self.config["eps_start"]
        eps_end = self.config["eps_end"]
        decay_steps = self.config["decay_steps"]

        # After decay_steps, stay at floor
        if step >= decay_steps:
            return eps_end

        # Linear interpolation: eps_start -> eps_end over decay_steps
        frac = step / decay_steps
        return eps_start + (eps_end - eps_start) * frac

    def select_action(
        self,
        step: int,
        q_values: np.ndarray,
        valid_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Epsilon-greedy: explore with probability epsilon."""
        eps = self.epsilon(step)

        # Explore: random valid action
        if rng.random() < eps:
            valid_actions = np.where(valid_mask > 0)[0]
            return int(rng.choice(valid_actions))

        # Greedy: argmax of valid Q-values (invalid actions set to -inf)
        masked_q = np.copy(q_values)
        masked_q[valid_mask == 0] = -np.inf
        return int(np.argmax(masked_q))


class ExponentialDecay(ExplorationStrategy):
    """Exponential epsilon decay: eps = max(eps_end, eps_start * decay_rate^step).

    Aggressive early reduction, asymptotic approach to floor.
    Action selection: standard epsilon-greedy.
    """

    def epsilon(self, step: int) -> float:
        """Exponential decay with floor."""
        eps_start = self.config["eps_start"]
        eps_end = self.config["eps_end"]
        decay_rate = self.config["decay_rate"]

        # Compute exponential decay: eps_start * decay_rate^step
        eps = eps_start * (decay_rate ** step)
        # Floor at eps_end to prevent epsilon from becoming arbitrarily small
        return max(eps_end, eps)

    def select_action(
        self,
        step: int,
        q_values: np.ndarray,
        valid_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Epsilon-greedy: explore with probability epsilon."""
        eps = self.epsilon(step)

        # Explore: random valid action with probability epsilon
        if rng.random() < eps:
            valid_actions = np.where(valid_mask > 0)[0]
            return int(rng.choice(valid_actions))

        # Greedy: argmax of valid Q-values (invalid actions set to -inf)
        masked_q = np.copy(q_values)
        masked_q[valid_mask == 0] = -np.inf
        return int(np.argmax(masked_q))


class LogarithmicDecay(ExplorationStrategy):
    """Logarithmic epsilon decay: fast early, asymptotic slowdown.

    Formula: eps = eps_end + (eps_start - eps_end) * (1 - log(step+1)/log(decay_steps+1))

    Provides middle ground between linear and exponential.
    Action selection: standard epsilon-greedy.
    """

    def epsilon(self, step: int) -> float:
        """Logarithmic decay with floor."""
        eps_start = self.config["eps_start"]
        eps_end = self.config["eps_end"]
        decay_steps = self.config["decay_steps"]

        # After decay_steps, stay at floor
        if step >= decay_steps:
            return eps_end

        # Log progress: 0 at step 0, 1 at decay_steps
        # Using log(step+1) to handle step=0 naturally
        progress = math.log(step + 1) / math.log(decay_steps + 1)
        eps = eps_end + (eps_start - eps_end) * (1.0 - progress)
        return eps

    def select_action(
        self,
        step: int,
        q_values: np.ndarray,
        valid_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Standard epsilon-greedy."""
        eps = self.epsilon(step)

        # Explore: random valid action with probability epsilon
        if rng.random() < eps:
            valid_actions = np.where(valid_mask > 0)[0]
            return int(rng.choice(valid_actions))

        # Greedy: argmax of valid Q-values (invalid actions set to -inf)
        masked_q = np.copy(q_values)
        masked_q[valid_mask == 0] = -np.inf
        return int(np.argmax(masked_q))
