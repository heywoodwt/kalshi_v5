"""Exploration strategies for DQN agent.

Provides abstract base class and concrete implementations for epsilon-greedy
and alternative exploration approaches. Each strategy controls both the
exploration rate (epsilon) and the action selection logic.
"""
from abc import ABC, abstractmethod
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
