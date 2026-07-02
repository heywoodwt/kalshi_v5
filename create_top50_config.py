"""Create Top 50 refined config from temporal split results."""
import re

# Read the markdown file
with open('hpc/mm_temporal_split_results.md', 'r') as f:
    content = f.read()

# Find the "All Categories" section
all_cat_match = re.search(r'## All Categories\n\n\| Category.*?\n\|---.*?\n(.*?)(?=\n##|\Z)', content, re.DOTALL)

if not all_cat_match:
    print("Could not find All Categories section")
    exit(1)

table_content = all_cat_match.group(1)
lines = table_content.strip().split('\n')

categories = []
for line in lines:
    if not line.strip() or not line.startswith('|'):
        continue

    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 11:
        continue

    try:
        cat_name = parts[1]
        train_pnl_str = parts[4].replace('$', '').replace('+', '').strip()
        test_pnl_str = parts[5].replace('$', '').replace('+', '').strip()
        overfit_str = parts[8].replace('x', '').strip()

        if not train_pnl_str or not test_pnl_str:
            continue

        train_pnl = float(train_pnl_str)
        test_pnl = float(test_pnl_str)

        # Parse overfit
        if 'inf' in overfit_str:
            overfit = float('inf')
        elif overfit_str.startswith('-'):
            overfit = 999
        else:
            overfit = abs(float(overfit_str))

        # Criteria: profitable on BOTH splits, overfit < 2x
        if train_pnl > 0 and test_pnl > 0 and overfit < 2.0:
            categories.append({
                'name': cat_name,
                'test_pnl': test_pnl,
                'train_pnl': train_pnl,
                'overfit': overfit
            })
    except (ValueError, IndexError):
        continue

# Sort by test PnL descending
categories.sort(key=lambda x: x['test_pnl'], reverse=True)

# Take top 50
top50 = categories[:50]

print(f"Found {len(categories)} consistent performers")
print(f"Taking top 50 by test PnL")
print()

# Create the config file
with open('rl_bot/live_config_top50.py', 'w') as f:
    f.write('"""\n')
    f.write('Top 50 Refined Configuration\n')
    f.write('Highest-performing categories from temporal split testing\n')
    f.write('\n')
    f.write('Selection: Top 50 by test PnL from 416 consistent performers\n')
    f.write('Capital: $95.37 / 50 = $1.91 per category (vs $0.35 for 262)\n')
    f.write('Benefits:\n')
    f.write('  - Higher capital per category overcomes fee drag\n')
    f.write('  - Concentrated on best performers only\n')
    f.write('  - Better position sizing for profit potential\n')
    f.write('"""\n\n')
    f.write('from dataclasses import dataclass\n')
    f.write('from typing import List\n\n')

    f.write('@dataclass\n')
    f.write('class CategoryConfig:\n')
    f.write('    """Configuration for a single category."""\n')
    f.write('    name: str\n')
    f.write('    max_contracts: int = 2  # Increased from 1\n')
    f.write('    max_inventory: int = 4  # Increased from 2\n')
    f.write('    capital_allocation: float = 1.90\n')
    f.write('    test_pnl: float = 0.0\n\n')

    f.write(f'# Top 50 performers (sorted by test PnL)\n')
    f.write('TOP_50_CATEGORIES = [\n')

    for i, cat in enumerate(top50, 1):
        f.write(f'    # {i}. Test: ${cat["test_pnl"]:.2f}, Train: ${cat["train_pnl"]:.2f}, Overfit: {cat["overfit"]:.2f}x\n')
        f.write(f'    CategoryConfig(\n')
        f.write(f'        name="{cat["name"]}",\n')
        f.write(f'        test_pnl={cat["test_pnl"]:.2f},\n')
        f.write(f'        capital_allocation=1.90\n')
        f.write(f'    ),\n')

    f.write(']\n\n')

    # Trading config
    f.write('# Calculate allocation\n')
    f.write('TOTAL_ALLOCATION = sum(cat.capital_allocation for cat in TOP_50_CATEGORIES)\n')
    f.write('RESERVE_CAPITAL = max(95.37 - TOTAL_ALLOCATION, 5.00)  # Keep at least $5 reserve\n\n')

    f.write('# Trading configuration\n')
    f.write('TRADING_CONFIG = {\n')
    f.write('    "mode": "live",\n')
    f.write('    "capital": 95.37,\n')
    f.write('    "reserve": RESERVE_CAPITAL,\n')
    f.write('    "active_capital": min(TOTAL_ALLOCATION, 90.37),\n\n')
    f.write('    # Global limits - Balanced\n')
    f.write('    "max_daily_loss": 12.00,\n')
    f.write('    "max_position_value": 70.00,\n')
    f.write('    "stop_loss_threshold": -20.00,\n\n')
    f.write('    # Execution - Subpenny enabled (as in testing)\n')
    f.write('    "subpenny_enabled": True,\n')
    f.write('    "quote_offset_bid": +0.001,\n')
    f.write('    "quote_offset_ask": -0.001,\n')
    f.write('    "quote_size": 2,  # Increased from 1\n\n')
    f.write('    # Risk management - Less conservative with higher capital\n')
    f.write('    "max_inventory_per_category": 4,  # Increased from 2\n')
    f.write('    "max_total_inventory": 30,  # Increased from 20\n')
    f.write('    "halt_on_consecutive_losses": 3,\n\n')
    f.write('    # Categories\n')
    f.write('    "categories": TOP_50_CATEGORIES,\n')
    f.write('}\n\n')

    # Monitoring
    f.write('# Monitoring configuration\n')
    f.write('MONITORING_CONFIG = {\n')
    f.write('    "alert_on_daily_loss": 8.00,\n')
    f.write('    "alert_on_stop_loss": -20.00,\n')
    f.write('    "alert_on_fill_rate_drop": 0.05,\n')
    f.write('    "alert_on_websocket_downtime": 60,\n\n')
    f.write('    # Logging\n')
    f.write('    "log_all_quotes": False,\n')
    f.write('    "log_all_fills": True,\n')
    f.write('    "log_all_orders": True,\n')
    f.write('    "log_pnl_updates": True,\n\n')
    f.write('    # Reports\n')
    f.write('    "daily_report_enabled": True,\n')
    f.write('    "hourly_summary_enabled": True,\n')
    f.write('}\n\n')

    f.write('if __name__ == "__main__":\n')
    f.write('    print("=" * 80)\n')
    f.write('    print("TOP 50 REFINED CONFIGURATION")\n')
    f.write('    print("=" * 80)\n')
    f.write('    print(f"Categories: {len(TOP_50_CATEGORIES)}")\n')
    f.write('    print(f"Capital per category: ${TOP_50_CATEGORIES[0].capital_allocation:.2f}")\n')
    f.write('    print(f"Total allocation: ${TOTAL_ALLOCATION:.2f}")\n')
    f.write('    print(f"Reserve: ${RESERVE_CAPITAL:.2f}")\n')
    f.write('    print()\n')
    f.write('    print("Top 10:")\n')
    f.write('    for i, cat in enumerate(TOP_50_CATEGORIES[:10], 1):\n')
    f.write('        print(f"{i:2}. {cat.name:<30} Test PnL: ${cat.test_pnl:>6.2f}")\n')
    f.write('    print()\n')

print("✓ Created rl_bot/live_config_top50.py")
print()
print("Top 10 by test PnL:")
for i, cat in enumerate(top50[:10], 1):
    print(f"{i:2}. {cat['name']:<30} Test: ${cat['test_pnl']:>6.2f}  Train: ${cat['train_pnl']:>6.2f}  Overfit: {cat['overfit']:.2f}x")
