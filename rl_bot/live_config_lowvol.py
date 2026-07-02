"""
Low-Volatility Deployment Configuration (Phase 2)

Deploy ONLY where the edge is proven. The 3-month analysis and the 2026-07-01
markout calibration (output/calibration_latest.csv) both show market making is
profitable on low-volatility, mean-reverting categories and loses on trending
ones. Whitelist (within-market price vol in parentheses):

    KXAAAGASM (0.011)  gas monthly  — near-zero adverse selection, +net live edge
    KXAAAGASD (~0.01)  gas daily    — +net live edge in calibration
    KXWCGAME  (0.019)  World Cup    — low vol
    KXADP     (0.021)  ADP payrolls — 94-100% one-way retail flow, +net live edge
    KXBTCD    (0.022)  BTC daily    — low vol, 20-dim retrained model available

Explicitly EXCLUDED: KXBTC (vol 0.070) and KXMLBGAME (vol 0.185) — adverse
selection eats the spread there; the +9,744 KXBTC eval was simulator bias.

Model requirement: the live trader validates every checkpoint is 20-dim and
DISABLES categories whose models don't match. Checkpoint search order per
category C: realistic_20dim_C_final.zip, then mm_C_C_final.zip (legacy 16-dim,
will be rejected by the dim check). As HPC retraining lands new
realistic_20dim_* files, categories activate automatically on restart.
"""

from dataclasses import dataclass


@dataclass
class CategoryConfig:
    """Configuration for a single category."""
    name: str
    max_contracts: int = 1     # minimum size for the 1-week validation run
    max_inventory: int = 5     # small book while validating eval-to-live
    capital_allocation: float = 19.00
    vol_3mo: float = 0.0       # within-market price volatility (3-month data)


# Whitelist: low-vol categories with demonstrated (or plausible) live edge.
# Out-of-sample eval scorecard (2026-07-01, S3 holdout, Phase 3 sim):
#   KXBTCD    +479.18 (262 eps)   KXWCGAME  +121.26 (43 eps, median +0.60)
#   KXAAAGASM  +21.38 (32 eps, variance-dominated — a few +/-10 thin-market eps)
#   KXADP      unevaluable (8 trades in holdout) — kept on live markout evidence
#   KXAAAGASD   -2.37 (4/26 winning eps) — REMOVED: model loses out-of-sample
LOWVOL_CATEGORIES = [
    CategoryConfig(name="KXAAAGASM", vol_3mo=0.011),
    CategoryConfig(name="KXWCGAME", vol_3mo=0.019),
    CategoryConfig(name="KXADP", vol_3mo=0.021),
    CategoryConfig(name="KXBTCD", vol_3mo=0.022),
]

# Trading configuration — deliberately tight limits for the validation week
TRADING_CONFIG = {
    "mode": "live",
    "capital": 95.00,

    # Global limits: small on purpose. The goal of this deployment is to
    # measure eval-to-live correlation, not to maximize PnL.
    "max_daily_loss": 5.00,
    "max_position_value": 40.00,
    "stop_loss_threshold": -10.00,

    # Execution
    "subpenny_enabled": True,

    # Risk management
    "halt_on_consecutive_losses": 3,

    # Categories
    "categories": LOWVOL_CATEGORIES,

    # 20-dim bias-corrected models (fall back to mm_* which the dim check rejects)
    "checkpoint_prefix": "realistic_20dim",
}

# Monitoring configuration — the weekly gates for scaling up:
#   1. taker fill fraction < 10% (hourly summary prints PASS/FAIL)
#   2. fees per fill match the maker schedule
#   3. live PnL >= 0 over the week
MONITORING_CONFIG = {
    "alert_on_daily_loss": 3.00,
    "alert_on_stop_loss": -10.00,
    "alert_on_fill_rate_drop": 0.05,
    "alert_on_websocket_downtime": 60,
    "log_all_quotes": False,
    "log_all_fills": True,
    "log_all_orders": True,
    "log_pnl_updates": True,
    "daily_report_enabled": True,
    "hourly_summary_enabled": True,
}
