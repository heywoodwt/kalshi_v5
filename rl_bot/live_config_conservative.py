"""
Conservative Deployment Configuration
Capital-constrained deployment for $95.82
Selected categories with manageable market counts
Total markets: ~75 (150 orders = $75 reserved capital)
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    max_contracts: int = 1
    max_inventory: int = 5
    capital_allocation: float = 10.00  # Higher allocation per category

# Conservative selection: 2 categories with 75 total markets
CONSERVATIVE_CATEGORIES = [
    "KXADP",         # 42 markets - Employment data
    "KXAAAGASM",     # 33 markets - Gas prices
]

# Create category configs
CONSERVATIVE_CATEGORY_CONFIGS = [
    CategoryConfig(name=cat, capital_allocation=10.00)
    for cat in CONSERVATIVE_CATEGORIES
]

# Calculate allocation
TOTAL_ALLOCATION = sum(cat.capital_allocation for cat in CONSERVATIVE_CATEGORY_CONFIGS)
RESERVE_CAPITAL = 95.82 - TOTAL_ALLOCATION

# Trading configuration
TRADING_CONFIG = {
    "mode": "live",
    "capital": 95.82,
    "reserve": RESERVE_CAPITAL,
    "active_capital": TOTAL_ALLOCATION,

    # Global limits
    "max_daily_loss": 10.00,
    "max_position_value": 50.00,  # Conservative limit
    "stop_loss_threshold": -25.00,

    # Execution
    "subpenny_enabled": True,
    "quote_offset_bid": +0.001,
    "quote_offset_ask": -0.001,
    "quote_size": 1,

    # Risk management
    "max_inventory_per_category": 5,
    "max_total_inventory": 20,  # Conservative for capital
    "halt_on_consecutive_losses": 3,

    # Categories
    "categories": CONSERVATIVE_CATEGORY_CONFIGS,
}

# Monitoring configuration
MONITORING_CONFIG = {
    "alert_on_daily_loss": 5.00,
    "alert_on_stop_loss": -25.00,
    "alert_on_fill_rate_drop": 0.05,
    "alert_on_websocket_downtime": 60,

    # Logging
    "log_all_quotes": False,
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,

    # Reports
    "daily_report_enabled": True,
    "hourly_summary_enabled": True,
}

if __name__ == "__main__":
    print("=" * 80)
    print("CONSERVATIVE DEPLOYMENT CONFIGURATION")
    print("=" * 80)
    print(f"Categories: {len(CONSERVATIVE_CATEGORY_CONFIGS)}")
    print(f"Estimated markets: ~75")
    print(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    print(f"Active: ${TRADING_CONFIG['active_capital']:.2f}")
    print(f"Reserve: ${TRADING_CONFIG['reserve']:.2f}")
    print()
