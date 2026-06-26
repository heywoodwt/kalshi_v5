"""Market-making environment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MMConfig:
    """Configuration for the MMEnv market-making environment."""

    # Position limits
    max_inventory: int = 100
    quote_size: int = 10

    # API configuration for live trading
    api_environment: str = "demo"  # "demo" or "production"
    api_base_url: str = ""  # Auto-set based on environment if empty
    ws_url: str = ""        # Auto-set based on environment if empty

    # Subpenny pricing
    subpenny_enabled: bool = True  # Enable/disable subpenny adjustment

    # Balance monitoring
    balance_check_interval_s: int = 60  # Check balance every N seconds
    min_balance_cents: float = 100.0    # Minimum balance to allow trading ($1.00)

    def __post_init__(self):
        """Auto-configure API URLs based on environment."""
        # Auto-configure api_base_url if not provided
        if not self.api_base_url:
            if self.api_environment == "production":
                object.__setattr__(self, 'api_base_url',
                    'https://external-api.kalshi.com/trade-api/v2')
            else:  # demo
                object.__setattr__(self, 'api_base_url',
                    'https://external-api.demo.kalshi.co/trade-api/v2')

        # Auto-configure ws_url if not provided
        if not self.ws_url:
            if self.api_environment == "production":
                object.__setattr__(self, 'ws_url',
                    'wss://external-api-ws.kalshi.com/trade-api/ws/v2')
            else:  # demo
                object.__setattr__(self, 'ws_url',
                    'wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2')
