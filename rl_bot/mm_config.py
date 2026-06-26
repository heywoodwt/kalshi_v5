"""Market-making environment configuration."""

from dataclasses import dataclass


@dataclass
class MMConfig:
    """Configuration for the MMEnv market-making environment."""

    # Position limits
    max_inventory: int = 100
    quote_size: int = 10

    # Placeholder for future hyperparameters
    pass
