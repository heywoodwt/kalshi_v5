"""Orderbook normalization and fillability helpers.

The Kalshi websocket orderbook payloads are intentionally kept lightweight here:
we only retain the top of book we need to decide whether a position can be
exited immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class OrderbookSnapshot:
    """Compact top-of-book snapshot used by the signal pipeline."""

    ticker: str
    yes_price: float | None = None
    yes_size: int = 0
    no_price: float | None = None
    no_size: int = 0
    updated_at: datetime | None = None

    def age_s(self, current_time: datetime | None = None) -> float | None:
        """Return book age in seconds, or None if no timestamp is available."""
        if self.updated_at is None:
            return None
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        return max(0.0, (current_time - self.updated_at).total_seconds())


def _coerce_float(value: Any) -> float | None:
    """Convert common Kalshi payload value shapes into a float price."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("price", "yes_price", "no_price", "value", "p"):
            if key in value:
                return _coerce_float(value[key])
    return None


def _coerce_int(value: Any) -> int:
    """Convert a level size/count payload into an integer size."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(round(value)))
    if isinstance(value, str):
        try:
            return max(0, int(round(float(value))))
        except ValueError:
            return 0
    if isinstance(value, dict):
        for key in ("count", "quantity", "size", "qty", "volume"):
            if key in value:
                return _coerce_int(value[key])
    return 0


def _first_level(levels: Any) -> Any:
    """Kalshi often sends ordered ladders; the first level is the top of book."""
    if levels is None:
        return None
    if isinstance(levels, list) or isinstance(levels, tuple):
        return levels[0] if levels else None
    if isinstance(levels, dict):
        # Some payloads use a single level object instead of a list.
        if any(key in levels for key in ("price", "count", "quantity", "size", "qty")):
            return levels
        # Fallback: if the mapping looks like a ladder, take the first item.
        for value in levels.values():
            return value
    return levels


def _normalize_side(levels: Any) -> tuple[float | None, int]:
    """Extract a top price and size from one side of the book."""
    level = _first_level(levels)
    if level is None:
        return None, 0

    price = _coerce_float(level)
    size = _coerce_int(level)
    return price, size


def extract_orderbook_snapshot(
    payload: dict[str, Any] | None,
    updated_at: datetime | None = None,
) -> OrderbookSnapshot | None:
    """Normalize a raw Kalshi orderbook message into a compact snapshot."""
    if not payload:
        return None

    data = payload.get("msg", payload)
    ticker = str(data.get("market_ticker") or data.get("ticker") or "")
    if not ticker:
        return None

    yes_price, yes_size = _normalize_side(data.get("yes"))
    no_price, no_size = _normalize_side(data.get("no"))

    if yes_price is None and no_price is None:
        return None

    if updated_at is None:
        updated_at = datetime.now(timezone.utc)

    return OrderbookSnapshot(
        ticker=ticker,
        yes_price=yes_price,
        yes_size=yes_size,
        no_price=no_price,
        no_size=no_size,
        updated_at=updated_at,
    )

