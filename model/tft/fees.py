"""Kalshi fee calculations (cents-based, for TFT model internals).

Fee formula (from Kalshi fee schedule):
    taker_fee = ceil(0.07 * contracts * price * (1 - price))
    maker_fee = ceil(0.0175 * contracts * price * (1 - price))

Where price is in dollars (0.00 to 1.00). Fees returned in cents.
"""

from __future__ import annotations

import math

# Kalshi fee rates
TAKER_RATE = 0.07
MAKER_RATE = 0.0175


def kalshi_taker_fee(price_cents: int, contracts: int = 1) -> int:
    """Compute taker fee in cents for a trade at the given price."""
    return _fee(price_cents, contracts, TAKER_RATE)


def kalshi_maker_fee(price_cents: int, contracts: int = 1) -> int:
    """Compute maker fee in cents for a trade at the given price."""
    return _fee(price_cents, contracts, MAKER_RATE)


def _fee(price_cents: int, contracts: int, rate: float) -> int:
    """Core fee calculation: ceil(rate * contracts * p * (1-p)), min 0."""
    p = price_cents / 100.0
    raw = rate * contracts * p * (1.0 - p)
    return max(0, math.ceil(raw))


def round_trip_fee(
    entry_price_cents: int,
    exit_price_cents: int,
    contracts: int = 1,
    maker: bool = False,
) -> int:
    """Total fee for entry + exit (both on same side)."""
    fee_fn = kalshi_maker_fee if maker else kalshi_taker_fee
    return fee_fn(entry_price_cents, contracts) + fee_fn(exit_price_cents, contracts)


def net_pnl(
    entry_price: int,
    exit_price: int,
    contracts: int = 1,
    maker: bool = False,
) -> int:
    """Net P&L in cents after subtracting round-trip fees."""
    gross = (exit_price - entry_price) * contracts
    fees = round_trip_fee(entry_price, exit_price, contracts, maker)
    return gross - fees