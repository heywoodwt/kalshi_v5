"""
Full Deployment Configuration
All 59 validated categories with open markets from HPC training
Capital: $95.82
Markets: 2,135 individual market tickers
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    max_contracts: int = 1
    max_inventory: int = 5
    capital_allocation: float = 1.50  # Equal allocation across all categories

# All 59 validated categories with open markets (from HPC sync)
FULL_DEPLOYMENT_CATEGORIES = [
    "KXAAAGASD",
    "KXAAAGASM",
    "KXAAAGASW",
    "KXAAPLCEOCHANGE",
    "KXAAPLPRICEFOLD",
    "KXABNB",
    "KXABRAHAMSA",
    "KXACQANNOUNCESPACEX",
    "KXADP",
    "KXAFLGAME",
    "KXAISPIKE",
    "KXAISTREAMSERIES",
    "KXALBUMDEBUT",
    "KXALBUMEQUIV",
    "KXALBUMLENGTH",
    "KXALBUMRELEASE",
    "KXALIENS",
    "KXALITOOUT",
    "KXALPRIMARY",
    "KXAPRPOTUS",
    "KXARODGRETIRE",
    "KXARREST",
    "KXARTISTSTREAMS",
    "KXARTISTSTREAMSU",
    "KXATPCHALLENGERMATCH",
    "KXATPMATCH",
    "KXATPSETWINNER",
    "KXAVGMEASLESDJT",
    "KXBALANCEPOWERCOMBO",
    "KXBALLONDOR",
    "KXBBCHARTPOSITIONALBUM",
    "KXBBCHARTPOSITIONSONG",
    "KXBESTLLMCHINA",
    "KXBILLS",
    "KXBILLSCOUNT",
    "KXBLUESPACEX",
    "KXBLUETSUNAMICOMBO",
    "KXBLUEWAVECOMBO",
    "KXBOND",
    "KXBOXING",
    "KXBRENTD",
    "KXBRENTMON",
    "KXBRENTW",
    "KXBRPRES",
    "KXBTC",
    "KXBTCD",
    "KXBTCMAXMON",
    "KXBTCMAXY",
    "KXBTCMINMON",
    "KXBTCMINY",
    "KXBTCVSGOLD",
    "KXBTCY",
    "KXCABILLIONAIRETAX",
    "KXCABLEAVE",
    "KXCABOUT",
    "KXLOLTOTALMAPS",
    "KXNBAMVP",
    "KXRAINLAXM",
    "KXSPOTIFYW",
]

# Create category configs with equal allocation
FULL_CATEGORIES = [
    CategoryConfig(name=cat) for cat in FULL_DEPLOYMENT_CATEGORIES
]

# Calculate allocation
TOTAL_ALLOCATION = sum(cat.capital_allocation for cat in FULL_CATEGORIES)
RESERVE_CAPITAL = 95.82 - min(TOTAL_ALLOCATION, 90.00)  # Keep at least $5 reserve

print(f"Total categories: {len(FULL_CATEGORIES)}")
print(f"Target allocation: ${min(TOTAL_ALLOCATION, 90.00):.2f}")
print(f"Reserve capital: ${RESERVE_CAPITAL:.2f}")

# Trading configuration
TRADING_CONFIG = {
    "mode": "live",
    "capital": 95.82,
    "reserve": RESERVE_CAPITAL,
    "active_capital": min(TOTAL_ALLOCATION, 90.00),

    # Global limits
    "max_daily_loss": 10.00,
    "max_position_value": 60.00,
    "stop_loss_threshold": -25.00,

    # Execution
    "subpenny_enabled": True,
    "quote_offset_bid": +0.001,
    "quote_offset_ask": -0.001,
    "quote_size": 1,

    # Risk management
    "max_inventory_per_category": 5,
    "max_total_inventory": 50,  # Increased for more markets
    "halt_on_consecutive_losses": 3,

    # Categories
    "categories": FULL_CATEGORIES,
}

# Monitoring configuration
MONITORING_CONFIG = {
    "alert_on_daily_loss": 5.00,
    "alert_on_stop_loss": -25.00,
    "alert_on_fill_rate_drop": 0.05,
    "alert_on_websocket_downtime": 60,

    # Logging
    "log_all_quotes": False,  # Too noisy with 2135 markets
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,

    # Reports
    "daily_report_enabled": True,
    "hourly_summary_enabled": True,
}

if __name__ == "__main__":
    print("=" * 80)
    print("FULL DEPLOYMENT CONFIGURATION")
    print("=" * 80)
    print(f"Categories: {len(FULL_CATEGORIES)}")
    print(f"Estimated markets: ~2,135")
    print(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    print(f"Active: ${TRADING_CONFIG['active_capital']:.2f}")
    print(f"Reserve: ${TRADING_CONFIG['reserve']:.2f}")
    print()
