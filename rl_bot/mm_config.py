"""Market-making environment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MMConfig:
    """Configuration for the MMEnv market-making environment."""

    # Position limits
    max_inventory: int = 20   # matches actual training runs
    quote_size: int = 1        # matches live trading (1 contract per quote)

    # API configuration for live trading
    api_environment: str = "demo"  # "demo" or "production"
    api_base_url: str = ""  # Auto-set based on environment if empty
    ws_url: str = ""        # Auto-set based on environment if empty

    # Inventory & risk
    inventory_lambda: float = 0.01  # base penalty per unit of inventory
    inventory_tte_scale: bool = True  # scale penalty by 1/sqrt(tte_h+1)

    # Fees (Kalshi)
    maker_fee_rate: float = 0.0175
    taker_fee_rate: float = 0.07     # Kalshi taker rate (paid when crossing the spread)
    # Fraction of sim fills charged the taker rate. 0.945 was the PRE-fix live
    # pathology (quotes crossed the book); with post-only quoting only the
    # stop-loss/expiry exits cross. Re-measure from the live taker fraction
    # (hourly summary) and update after each deployment week.
    taker_fill_prob: float = 0.10

    # Subpenny pricing
    subpenny_enabled: bool = True  # Enable/disable subpenny adjustment

    # Domain randomization (training only — makes agent robust to illiquid conditions)
    domain_rand_spread_prob: float = 0.15   # probability per step of injecting extra spread
    domain_rand_spread_max: float = 0.04    # max extra spread to inject (1-4 cents uniform)
    domain_rand_volume_prob: float = 0.10   # probability per step of deleting top-of-book volume
    through_fill_haircut: float = 0.33      # fraction of through-fills actually received (~1/3: realistic queue position)

    # Live deployment risk filter
    vol_filter_threshold: float = 0.05  # skip quoting when rolling mid-price vol exceeds this (MM loses in trending markets)
    min_quote_edge: float = 0.01  # required per-contract profit beyond round-trip ceil'd fees before quoting

    # Live capital controls. A two-sided 1-lot quote locks ~$1 of collateral
    # (bid price + 1 - ask price), so quoting 310 markets with $95 exhausted
    # the balance instantly (observed: 4,900+ insufficient_balance rejections).
    max_open_orders: int = 60           # global cap on concurrent resting orders (~$30-50 locked)
    balance_backoff_s: float = 60.0     # pause all quoting this long after an insufficient_balance rejection
    quote_band_lo: float = 0.05         # don't quote when mid is outside [lo, hi]: extreme-priced
    quote_band_hi: float = 0.95         # contracts tie up collateral for pennies of edge near settlement

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
