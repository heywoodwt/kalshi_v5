"""KXBTCD paper-trading configuration for the 20-dim bias-corrected model.

Loads rl_bot/mm_checkpoints/realistic_20dim_KXBTCD_final.zip (checkpoint_prefix
"realistic_20dim" + category name "KXBTCD"). Single category, conservative demo
risk limits. Intended to run with PAPER_MODE=true.
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


# KXBTCD only (BTC daily — low-vol, MM-friendly)
CATEGORIES = [
    CategoryConfig(
        name="KXBTCD",
        max_inventory=6,
        capital_allocation=50.00,
    ),
]

TRADING_CONFIG = {
    "mode": "demo",
    "capital": 50.00,
    "reserve": 5.00,
    "active_capital": 45.00,

    # Global risk limits — conservative for paper trading
    "max_daily_loss": 10.00,
    "max_position_value": 40.00,
    "stop_loss_threshold": -15.00,

    # Execution — subpenny left on (metadata loader gates it per ticker/tick size)
    "subpenny_enabled": True,
    "quote_offset_bid": 0.0,
    "quote_offset_ask": 0.0,
    "quote_size": 1,

    # Risk management
    "max_inventory_per_category": 6,
    "max_total_inventory": 10,
    "halt_on_consecutive_losses": 3,

    # Load realistic_20dim_KXBTCD_final.zip
    "categories": CATEGORIES,
    "checkpoint_prefix": "realistic_20dim",
}

MONITORING_CONFIG = {
    "alert_on_daily_loss": 8.00,
    "alert_on_stop_loss": -15.00,
    "alert_on_fill_rate_drop": 0.05,
    "alert_on_websocket_downtime": 60,

    "log_all_quotes": True,
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,

    "daily_report_enabled": True,
    "hourly_summary_enabled": True,
}
