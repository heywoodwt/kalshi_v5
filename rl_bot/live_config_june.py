"""
June 2026 Live Trading Configuration
Based on actual June 29-30 trading data from S3

Top 5 most active categories by June trade volume:
1. KXBTC: 84,839 trades
2. KXWCADVANCE: 68,295 trades (World Cup advance)
3. KXBTCD: 57,284 trades (Bitcoin daily)
4. KXWCGAME: 23,280 trades (World Cup games)
5. KXMLBGAME: 21,417 trades (MLB games)

Capital: $95.37 / 5 = $19.07 per category
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    max_contracts: int = 3  # Higher per category with fewer total
    max_inventory: int = 6   # Increased inventory limit
    capital_allocation: float = 19.07
    june_trades: int = 0

# Top 5 June categories with trained models
JUNE_TOP5_CATEGORIES = [
    CategoryConfig(
        name="KXBTC",
        june_trades=84839,
        capital_allocation=19.07
    ),
    CategoryConfig(
        name="KXWCADVANCE",
        june_trades=68295,
        capital_allocation=19.07
    ),
    CategoryConfig(
        name="KXBTCD",
        june_trades=57284,
        capital_allocation=19.07
    ),
    CategoryConfig(
        name="KXWCGAME",
        june_trades=23280,
        capital_allocation=19.07
    ),
    CategoryConfig(
        name="KXMLBGAME",
        june_trades=21417,
        capital_allocation=19.07
    ),
]

# Calculate allocation
TOTAL_ALLOCATION = sum(cat.capital_allocation for cat in JUNE_TOP5_CATEGORIES)
RESERVE_CAPITAL = max(95.37 - TOTAL_ALLOCATION, 5.00)

# Trading configuration
TRADING_CONFIG = {
    "mode": "live",
    "capital": 95.37,
    "reserve": RESERVE_CAPITAL,
    "active_capital": min(TOTAL_ALLOCATION, 90.37),

    # Global limits
    "max_daily_loss": 15.00,
    "max_position_value": 75.00,
    "stop_loss_threshold": -25.00,

    # Execution - Subpenny enabled
    "subpenny_enabled": True,
    "quote_offset_bid": +0.001,
    "quote_offset_ask": -0.001,
    "quote_size": 3,  # Higher quote size with more capital per category

    # Risk management
    "max_inventory_per_category": 6,  # Increased
    "max_total_inventory": 20,
    "halt_on_consecutive_losses": 3,

    # Categories
    "categories": JUNE_TOP5_CATEGORIES,

    # June-specific checkpoint naming
    "checkpoint_prefix": "june",  # Models named: june_KXBTC_final.zip
}

# Monitoring configuration
MONITORING_CONFIG = {
    "alert_on_daily_loss": 10.00,
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
    print("JUNE 2026 TRADING CONFIGURATION")
    print("=" * 80)
    print(f"Categories: {len(JUNE_TOP5_CATEGORIES)}")
    print(f"Capital per category: ${JUNE_TOP5_CATEGORIES[0].capital_allocation:.2f}")
    print(f"Total allocation: ${TOTAL_ALLOCATION:.2f}")
    print(f"Reserve: ${RESERVE_CAPITAL:.2f}")
    print()
    print("Active categories (by June trade volume):")
    for i, cat in enumerate(JUNE_TOP5_CATEGORIES, 1):
        print(f"{i}. {cat.name:<15} {cat.june_trades:>6,} June trades")
    print()
    print("Models: rl_bot/mm_checkpoints/june_{CATEGORY}_final.zip")
    print("Data: June 29-30, 2026 (553k trades)")
    print()
