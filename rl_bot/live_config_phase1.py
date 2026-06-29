"""
Phase 1 Live Trading Configuration
Based on temporal split validation (hpc/mm_temporal_split_results.md)
Capital: $95.82
Markets: 262/419 validated categories with open markets on Kalshi
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    test_pnl_per_episode: float
    overfit_ratio: float
    test_win_rate: float
    max_contracts: int
    max_inventory: int
    capital_allocation: float  # Dollars allocated to this category


# Top 10 validated categories with best test performance and low overfitting
# Selected from 262 live markets in temporal split results
PHASE1_CATEGORIES = [
    CategoryConfig(
        name="KXLOLTOTALMAPS",
        test_pnl_per_episode=14.14,
        overfit_ratio=0.09,  # Excellent - almost no overfitting!
        test_win_rate=1.00,
        max_contracts=1,
        max_inventory=3,
        capital_allocation=12.00,
    ),
    CategoryConfig(
        name="KXBBCHARTPOSITIONALBUM",
        test_pnl_per_episode=10.10,
        overfit_ratio=1.20,  # Acceptable
        test_win_rate=1.00,
        max_contracts=1,
        max_inventory=3,
        capital_allocation=10.00,
    ),
    CategoryConfig(
        name="KXRAINLAXM",
        test_pnl_per_episode=9.81,
        overfit_ratio=1.00,  # Perfect - train = test
        test_win_rate=1.00,
        max_contracts=1,
        max_inventory=3,
        capital_allocation=10.00,
    ),
    CategoryConfig(
        name="KXNBAMVP",
        test_pnl_per_episode=9.79,
        overfit_ratio=1.00,
        test_win_rate=1.00,
        max_contracts=1,
        max_inventory=3,
        capital_allocation=10.00,
    ),
    CategoryConfig(
        name="KXSPOTIFYW",
        test_pnl_per_episode=9.79,
        overfit_ratio=1.00,
        test_win_rate=1.00,
        max_contracts=1,
        max_inventory=3,
        capital_allocation=10.00,
    ),
    CategoryConfig(
        name="KXAFCCLGAME",
        test_pnl_per_episode=3.46,
        overfit_ratio=-0.60,  # Reverse overfitting - excellent!
        test_win_rate=0.67,
        max_contracts=1,
        max_inventory=5,
        capital_allocation=15.00,
    ),
    CategoryConfig(
        name="KXATPMATCH",
        test_pnl_per_episode=3.00,
        overfit_ratio=-4.50,  # Reverse overfitting - excellent!
        test_win_rate=0.33,
        max_contracts=1,
        max_inventory=5,
        capital_allocation=13.00,
    ),
    CategoryConfig(
        name="KXATPCHALLENGERMATCH",
        test_pnl_per_episode=1.28,
        overfit_ratio=1.40,
        test_win_rate=0.33,
        max_contracts=1,
        max_inventory=5,
        capital_allocation=10.00,
    ),
]

# Calculate total allocation
TOTAL_ALLOCATION = sum(cat.capital_allocation for cat in PHASE1_CATEGORIES)
RESERVE_CAPITAL = 95.82 - TOTAL_ALLOCATION

assert TOTAL_ALLOCATION <= 95.82, f"Over-allocated: ${TOTAL_ALLOCATION} > $95.82"
assert RESERVE_CAPITAL >= 0, f"Negative reserve: ${RESERVE_CAPITAL}"

print(f"Total allocation: ${TOTAL_ALLOCATION:.2f}")
print(f"Reserve capital: ${RESERVE_CAPITAL:.2f}")


# Trading configuration
TRADING_CONFIG = {
    "mode": "live",  # Phase 1 = live trading
    "capital": 95.82,
    "reserve": RESERVE_CAPITAL,
    "active_capital": TOTAL_ALLOCATION,

    # Global limits
    "max_daily_loss": 10.00,  # 10.4% of capital
    "max_position_value": 60.00,  # 62% of capital across all categories
    "stop_loss_threshold": -25.00,  # 26% drawdown = halt all trading

    # Execution
    "subpenny_enabled": True,  # Queue priority strategy
    "quote_offset_bid": +0.001,  # Subpenny bid for priority
    "quote_offset_ask": -0.001,  # Subpenny ask for priority
    "quote_size": 1,  # Single contract quotes

    # Risk management
    "max_inventory_per_category": 5,  # Per-category position limit
    "max_total_inventory": 20,  # Total across all categories
    "halt_on_consecutive_losses": 3,  # Pause category after 3 losses in a row

    # Categories
    "categories": PHASE1_CATEGORIES,
}


# Monitoring configuration
MONITORING_CONFIG = {
    "alert_on_daily_loss": 5.00,  # Alert at $5 loss (50% of daily limit)
    "alert_on_stop_loss": -25.00,
    "alert_on_fill_rate_drop": 0.05,  # Alert if fill rate < 5%
    "alert_on_websocket_downtime": 60,  # Alert if WS down > 60 seconds

    # Logging
    "log_all_quotes": True,
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,

    # Reports
    "daily_report_enabled": True,
    "hourly_summary_enabled": True,
}


# Expected performance (conservative 15% fill rate)
def calculate_expected_performance():
    """Calculate expected daily PnL based on backtest and assumed fill rate."""

    # Assume average episodes per day (conservative estimates)
    episodes_per_day = {
        "KXLOLTOTALMAPS": 0.5,  # May have low frequency
        "KXBBCHARTPOSITIONALBUM": 1.0,
        "KXRAINLAXM": 0.5,
        "KXNBAMVP": 0.3,  # Seasonal/infrequent
        "KXSPOTIFYW": 1.0,
        "KXAFCCLGAME": 3.7,  # From earlier analysis
        "KXATPMATCH": 8.5,  # High frequency tennis
        "KXATPCHALLENGERMATCH": 5.4,  # High frequency tennis
    }

    fill_rate = 0.15  # Conservative 15%

    total_daily_pnl_backtest = 0
    total_daily_pnl_expected = 0

    print("\nExpected Performance (15% Fill Rate):\n")
    print(f"{'Category':<25} {'Test PnL/ep':<12} {'Ep/day':<8} {'Backtest $/day':<15} {'Expected $/day':<15}")
    print("-" * 85)

    for cat in PHASE1_CATEGORIES:
        ep_day = episodes_per_day.get(cat.name, 1.0)
        backtest_daily = cat.test_pnl_per_episode * ep_day
        expected_daily = backtest_daily * fill_rate

        total_daily_pnl_backtest += backtest_daily
        total_daily_pnl_expected += expected_daily

        print(f"{cat.name:<25} ${cat.test_pnl_per_episode:<11.2f} {ep_day:<8.1f} ${backtest_daily:<14.2f} ${expected_daily:<14.2f}")

    print("-" * 85)
    print(f"{'TOTAL':<25} {'':<12} {'':<8} ${total_daily_pnl_backtest:<14.2f} ${total_daily_pnl_expected:<14.2f}")
    print()
    annual_pnl = total_daily_pnl_expected * 365
    roi = (annual_pnl / 95.82) * 100
    print(f"Monthly: ${total_daily_pnl_expected * 30:.2f}")
    print(f"Annual: ${annual_pnl:.2f}")
    print(f"ROI: {roi:.1f}% (${annual_pnl:.2f} profit on $95.82 capital)")
    print()

    return total_daily_pnl_expected


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 1 LIVE TRADING CONFIGURATION")
    print("=" * 80)
    print()
    print(f"Capital: ${TRADING_CONFIG['capital']:.2f}")
    print(f"Active: ${TRADING_CONFIG['active_capital']:.2f}")
    print(f"Reserve: ${TRADING_CONFIG['reserve']:.2f}")
    print(f"Categories: {len(PHASE1_CATEGORIES)}")
    print()

    print("Category Allocations:")
    print("-" * 80)
    for cat in PHASE1_CATEGORIES:
        print(f"{cat.name:<25} ${cat.capital_allocation:<8.2f}  "
              f"Test: ${cat.test_pnl_per_episode:<7.2f}  "
              f"Overfit: {cat.overfit_ratio:<6.2f}x  "
              f"Win%: {cat.test_win_rate*100:<5.0f}%")
    print()

    expected_daily = calculate_expected_performance()

    print("Risk Limits:")
    print(f"  Max daily loss: ${TRADING_CONFIG['max_daily_loss']:.2f}")
    print(f"  Stop loss threshold: ${TRADING_CONFIG['stop_loss_threshold']:.2f}")
    print(f"  Max position value: ${TRADING_CONFIG['max_position_value']:.2f}")
    print()
