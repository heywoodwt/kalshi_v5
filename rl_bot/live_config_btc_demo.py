"""
BTC-Only Demo Trading Configuration
gamma=0.90 model, subpenny disabled (BTC uses 0.1c tick natively)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    max_contracts: int = 3
    max_inventory: int = 6
    capital_allocation: float = 50.00
    june_trades: int = 0


# BTC only
CATEGORIES = [
    CategoryConfig(
        name="KXBTC",
        june_trades=84839,
        capital_allocation=50.00,
    ),
]

# Trading configuration
TRADING_CONFIG = {
    "mode": "demo",
    "capital": 50.00,
    "reserve": 5.00,
    "active_capital": 45.00,

    # Global limits — conservative for demo
    "max_daily_loss": 10.00,
    "max_position_value": 40.00,
    "stop_loss_threshold": -15.00,

    # Execution — subpenny OFF (BTC tick is already 0.1c)
    "subpenny_enabled": False,
    "quote_offset_bid": 0.0,
    "quote_offset_ask": 0.0,
    "quote_size": 1,

    # Risk management
    "max_inventory_per_category": 6,
    "max_total_inventory": 10,
    "halt_on_consecutive_losses": 3,

    # Checkpoint — gamma=0.90 model
    "categories": CATEGORIES,
    "checkpoint_prefix": "june",  # loads june_KXBTC_final.zip (gamma=0.90)
}

# Monitoring configuration
MONITORING_CONFIG = {
    "alert_on_daily_loss": 8.00,
    "alert_on_stop_loss": -15.00,
    "alert_on_fill_rate_drop": 0.05,
    "alert_on_websocket_downtime": 60,

    # Logging — verbose for demo
    "log_all_quotes": True,
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,

    # Reports
    "daily_report_enabled": True,
    "hourly_summary_enabled": True,
}