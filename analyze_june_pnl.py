"""PnL analysis: replay June-trained models on real June 29-30 trade data.

Loads each june_{CATEGORY}_final.zip model and runs it through the MMEnv
using real trades from output/june_trades/. Reports per-category and
aggregate PnL, fill rates, and inventory stats.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

from rl_bot.mm_config import MMConfig
from rl_bot.mm_env import MMEnv, preprocess_mm_data
from rl_bot.mm_metadata import MarketMetadataLoader

# ---------- CONFIG ----------
JUNE_CATEGORIES = ["KXBTC", "KXWCADVANCE", "KXBTCD", "KXWCGAME", "KXMLBGAME"]
CHECKPOINT_DIR = Path("rl_bot/mm_checkpoints")
TRADES_DIR = Path("output/june_trades")
MARKETS_PATH = Path("output/rl_all_markets_3mo.parquet")
FEE_RATE = 0.0175  # Kalshi maker fee

# ---------- LOAD TRADES ----------
print("=" * 80)
print("JUNE 2026 PnL ANALYSIS — REPLAY ON REAL TRADE DATA")
print("=" * 80)

trade_files = sorted(TRADES_DIR.glob("*.parquet"))
if not trade_files:
    print("ERROR: No trade files in output/june_trades/")
    sys.exit(1)

all_trades = pl.concat([pl.read_parquet(f) for f in trade_files])
print(f"Loaded {len(all_trades):,} trades from {len(trade_files)} files")
print(f"Date range: {all_trades['created_time'].min()} → {all_trades['created_time'].max()}")

# Extract category prefix (everything before first dash)
cat_trades = all_trades.with_columns(
    pl.col("ticker").str.split("-").list.first().alias("category")
)
cat_summary = cat_trades.group_by("category").agg(
    pl.len().alias("trades"),
    pl.col("ticker").n_unique().alias("tickers"),
).sort("trades", descending=True)

print(f"\nTop categories by trade count:")
for row in cat_summary.head(10).iter_rows(named=True):
    marker = " ← MODEL" if row["category"] in JUNE_CATEGORIES else ""
    print(f"  {row['category']:<30s} {row['trades']:>8,} trades  {row['tickers']:>4} tickers{marker}")

# ---------- LOAD METADATA ----------
metadata_loader = None
if MARKETS_PATH.exists():
    metadata_loader = MarketMetadataLoader(mode="parquet", parquet_path=str(MARKETS_PATH))
    print(f"\nLoaded market metadata from {MARKETS_PATH}")
else:
    print(f"\nWARNING: {MARKETS_PATH} not found — subpenny/metadata disabled")

# ---------- RUN EVAL PER CATEGORY ----------
config = MMConfig(
    max_inventory=20,
    quote_size=1,
    subpenny_enabled=True,
    api_environment="demo",
)

all_results = []

for cat_name in JUNE_CATEGORIES:
    model_path = CHECKPOINT_DIR / f"june_{cat_name}_final.zip"
    if not model_path.exists():
        print(f"\n--- {cat_name}: SKIPPED (no checkpoint) ---")
        continue

    # Filter trades for this category
    cat_df = all_trades.filter(pl.col("ticker").str.starts_with(cat_name + "-"))
    if len(cat_df) == 0:
        # Some categories like KXBTC match KXBTC15M too — try exact prefix
        cat_df = all_trades.filter(
            pl.col("ticker").str.split("-").list.first() == cat_name
        )
    if len(cat_df) == 0:
        print(f"\n--- {cat_name}: SKIPPED (no trades) ---")
        continue

    # Preprocess into windows (no split — use all data for replay)
    windows = preprocess_mm_data(cat_df)
    total_windows = sum(len(v) for v in windows.values())
    n_tickers = len(windows)

    if total_windows == 0:
        print(f"\n--- {cat_name}: SKIPPED (no windows after preprocessing) ---")
        continue

    # Load model
    model = PPO.load(str(model_path), env=None)

    # Create env and run episodes
    env = MMEnv(ticker_data=windows, cfg=config, metadata_loader=metadata_loader)

    cat_results = []
    for ticker in sorted(windows.keys()):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        cat_results.append({
            "category": cat_name,
            "ticker": ticker,
            "steps": steps,
            "total_reward": float(episode_reward),
            "realized_pnl": float(info.get("realized_pnl", 0.0)),
            "unrealized_pnl": float(info.get("unrealized_pnl", 0.0)),
            "total_pnl": float(info.get("pnl", 0.0)),
            "final_inventory": int(info.get("inventory", 0)),
            "fills_buy": int(info.get("fills_buy", 0)),
            "fills_sell": int(info.get("fills_sell", 0)),
        })

    cat_df_result = pl.DataFrame(cat_results)
    all_results.extend(cat_results)

    # Category summary
    n_eps = len(cat_df_result)
    total_pnl = cat_df_result["total_pnl"].sum()
    realized = cat_df_result["realized_pnl"].sum()
    unrealized = cat_df_result["unrealized_pnl"].sum()
    mean_reward = cat_df_result["total_reward"].mean()
    win_rate = (cat_df_result["total_pnl"] > 0).sum() / max(n_eps, 1)
    total_fills = cat_df_result["fills_buy"].sum() + cat_df_result["fills_sell"].sum()
    total_steps = cat_df_result["steps"].sum()
    avg_inv = cat_df_result["final_inventory"].mean()

    # Estimate fees: fee_rate * contracts * price * (1-price)
    # Approximate avg price = 0.5 for worst-case fee estimate
    est_fees = FEE_RATE * total_fills * 0.25  # 0.5 * (1-0.5) = 0.25

    print(f"\n{'=' * 80}")
    print(f"  {cat_name}")
    print(f"{'=' * 80}")
    print(f"  Tickers:       {n_tickers:>6}    ({len(cat_df):,} trades)")
    print(f"  Episodes:      {n_eps:>6}    ({total_steps:,} steps)")
    print(f"  Total fills:   {total_fills:>6}    (buy={cat_df_result['fills_buy'].sum()}, sell={cat_df_result['fills_sell'].sum()})")
    print(f"  Fill rate:     {total_fills / max(total_steps * 2, 1):.1%}")
    print(f"  ---")
    print(f"  Realized PnL:  ${realized:>+10.4f}")
    print(f"  Unrealized:    ${unrealized:>+10.4f}")
    print(f"  Total PnL:     ${total_pnl:>+10.4f}")
    print(f"  Est. fees:     ${est_fees:>10.4f}")
    print(f"  Net (after fees): ${total_pnl - est_fees:>+10.4f}")
    print(f"  ---")
    print(f"  Mean reward:   {mean_reward:>+10.4f}")
    print(f"  Win rate:      {win_rate:>10.1%}")
    print(f"  Avg final inv: {avg_inv:>+10.1f}")

    # Top/bottom tickers
    sorted_by_pnl = cat_df_result.sort("total_pnl", descending=True)
    print(f"\n  Top 3 tickers:")
    for row in sorted_by_pnl.head(3).iter_rows(named=True):
        print(f"    {row['ticker']:<40s}  pnl=${row['total_pnl']:+.4f}  fills={row['fills_buy']+row['fills_sell']}  inv={row['final_inventory']:+d}")
    print(f"  Bottom 3 tickers:")
    for row in sorted_by_pnl.tail(3).iter_rows(named=True):
        print(f"    {row['ticker']:<40s}  pnl=${row['total_pnl']:+.4f}  fills={row['fills_buy']+row['fills_sell']}  inv={row['final_inventory']:+d}")

# ---------- AGGREGATE ----------
if all_results:
    results_df = pl.DataFrame(all_results)

    print(f"\n{'=' * 80}")
    print(f"  AGGREGATE — ALL CATEGORIES")
    print(f"{'=' * 80}")

    n_total = len(results_df)
    agg_pnl = results_df["total_pnl"].sum()
    agg_realized = results_df["realized_pnl"].sum()
    agg_unrealized = results_df["unrealized_pnl"].sum()
    agg_reward = results_df["total_reward"].sum()
    agg_fills = results_df["fills_buy"].sum() + results_df["fills_sell"].sum()
    agg_steps = results_df["steps"].sum()
    agg_win = (results_df["total_pnl"] > 0).sum() / max(n_total, 1)
    agg_fees = FEE_RATE * agg_fills * 0.25

    # Per-category breakdown
    per_cat = results_df.group_by("category").agg([
        pl.col("total_pnl").sum().alias("pnl"),
        pl.col("realized_pnl").sum().alias("realized"),
        (pl.col("fills_buy") + pl.col("fills_sell")).sum().alias("fills"),
        pl.len().alias("tickers"),
        (pl.col("total_pnl") > 0).sum().alias("wins"),
    ]).sort("pnl", descending=True)

    print(f"\n  Category breakdown:")
    print(f"  {'Category':<20s} {'PnL':>10s} {'Realized':>10s} {'Fills':>7s} {'Tickers':>8s} {'WinRate':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*7} {'-'*8} {'-'*8}")
    for row in per_cat.iter_rows(named=True):
        wr = row["wins"] / max(row["tickers"], 1)
        print(f"  {row['category']:<20s} ${row['pnl']:>+9.4f} ${row['realized']:>+9.4f} {row['fills']:>7,} {row['tickers']:>8} {wr:>7.1%}")

    print(f"\n  Totals:")
    print(f"    Episodes:      {n_total}")
    print(f"    Total steps:   {agg_steps:,}")
    print(f"    Total fills:   {agg_fills:,}")
    print(f"    ---")
    print(f"    Realized PnL:  ${agg_realized:>+.4f}")
    print(f"    Unrealized:    ${agg_unrealized:>+.4f}")
    print(f"    Gross PnL:     ${agg_pnl:>+.4f}")
    print(f"    Est. fees:     ${agg_fees:>.4f}")
    print(f"    Net PnL:       ${agg_pnl - agg_fees:>+.4f}")
    print(f"    ---")
    print(f"    Win rate:      {agg_win:.1%}")
    print(f"    Total reward:  {agg_reward:+.4f}")
    print(f"    Avg reward:    {agg_reward / max(n_total, 1):+.4f}")

    # Save detailed results
    out_path = "output/june_pnl_analysis.csv"
    results_df.write_csv(out_path)
    print(f"\n  Detailed results saved to {out_path}")

print(f"\n{'=' * 80}")
print("DONE")
print(f"{'=' * 80}")