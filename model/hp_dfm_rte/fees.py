"""Kalshi fee calculations.

Fee formula (from Kalshi fee schedule):
    taker_fee = ceil(0.07 * contracts * price * (1 - price))
    maker_fee = ceil(0.0175 * contracts * price * (1 - price))

Where price is in dollars (0.00 to 1.00).  Fees are in cents.
"""

from __future__ import annotations

import math

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
    """Net P&L in cents after subtracting round-trip fees.

    Gross P&L = (exit_price - entry_price) * contracts
    Net  P&L  = gross - round_trip_fee
    """
    gross = (exit_price - entry_price) * contracts
    fees = round_trip_fee(entry_price, exit_price, contracts, maker)
    return gross - fees


def gross_pnl(entry_price: int, exit_price: int, contracts: int = 1) -> int:
    """Gross P&L in cents (no fees)."""
    return (exit_price - entry_price) * contracts


def is_profitable_after_fees(
    expected_move_cents: float,
    price_cents: int,
    contracts: int = 1,
    maker: bool = False,
) -> bool:
    """Check if an expected move is large enough to overcome round-trip fees."""
    exit_price = int(round(min(99, max(1, price_cents + expected_move_cents))))
    fees = round_trip_fee(price_cents, exit_price, contracts, maker)
    return expected_move_cents > fees