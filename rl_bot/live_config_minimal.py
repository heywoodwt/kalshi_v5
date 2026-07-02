"""
Minimal Deployment Configuration
Ultra-conservative for exhausted capital
1 category with ~42 markets = ~84 orders × $0.25 avg = $21 capital needed
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    max_contracts: int = 1
    max_inventory: int = 3  # Reduced from 5
    capital_allocation: float = 15.00

# Minimal selection: 1 category
MINIMAL_CATEGORIES = [
    "KXADP",  # 42 markets - Employment data (most liquid)
]

# Create category configs
MINIMAL_CATEGORY_CONFIGS = [
    CategoryConfig(name=cat, capital_allocation=15.00, max_inventory=3)
    for cat in MINIMAL_CATEGORIES
]

# Calculate allocation
TOTAL_ALLOCATION = sum(cat.capital_allocation for cat in MINIMAL_CATEGORY_CONFIGS)
RESERVE_CAPITAL = 95.82 - TOTAL_ALLOCATION

# Trading configuration
TRADING_CONFIG = {
    "mode": "live",
    "capital": 95.82,
    "reserve": RESERVE_CAPITAL,
    "active_capital": TOTAL_ALLOCATION,

    # Global limits - VERY conservative
    "max_daily_loss": 5.00,
    "max_position_value": 30.00,  # Very conservative
    "stop_loss_threshold": -10.00,

    # Execution
    "subpenny_enabled": True,
    "quote_offset_bid": +0.001,
    "quote_offset_ask": -0.001,
    "quote_size": 1,

    # Risk management - VERY conservative
    "max_inventory_per_category": 3,
    "max_total_inventory": 10,  # Minimal exposure
    "halt_on_consecutive_losses": 2,

    # Categories
    "categories": MINIMAL_CATEGORY_CONFIGS,
}

# Monitoring configuration
MONITORING_CONFIG = {
    "alert_on_daily_loss": 3.00,
    "alert_on_stop_loss": -10.00,
    "alert_on_fill_rate_drop": 0.05,
    "alert_on_websocket_downtime": 60,

    # Logging
    "log_all_quotes": False,
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,

    # Reports
    "daily_report_enabled": True,
    "hourly_summary_enabled": False,  # Disabled for minimal config
}

if __name__ == "__main__":
    print("=" * 80)
    print("MINIMAL DEPLOYMENT CONFIGURATION")
    print("=" * 80)
    print(f"Categories: {len(MINIMAL_CATEGORY_CONFIGS)}")
    print(f"Estimated markets: ~42")
    print(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    print(f"Active: ${TRADING_CONFIG['active_capital']:.2f}")
    print(f"Reserve: ${TRADING_CONFIG['reserve']:.2f}")
    print()
