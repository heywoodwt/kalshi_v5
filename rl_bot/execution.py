"""Execution cost modeling for realistic RL trading simulation.

Estimates spread, slippage, fill probability, and fee classification
so the replay doesn't assume free, instant, guaranteed fills at mid-price.
All functions are O(1) — no loops, no allocations beyond the result struct.

When a fresh orderbook snapshot is available (live mode), real bid/ask
quotes replace the heuristic spread/depth estimates. Replay mode has no
orderbook data, so the estimation path runs automatically via the default
``orderbook=None``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.hp_dfm_rte.orderbook import OrderbookSnapshot

# ── Orderbook freshness ─────────────────────────────────────────────
# Max age before we fall back to heuristic estimation.
_ORDERBOOK_MAX_AGE_S = 30.0


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable result of compute_execution()."""
    exec_price: float    # price the trade actually fills at
    fee_rate: float      # maker (0.0175) or taker (0.07)
    fill_prob: float     # probability the order fills [0.05, 1.0]
    spread: float        # estimated bid-ask spread
    slippage: float      # price impact from walking the book


# ── Orderbook helpers ────────────────────────────────────────────────

def _orderbook_is_fresh(orderbook: OrderbookSnapshot | None) -> bool:
    """Return True when the orderbook has both sides quoted and is recent.

    Checks: not None, yes_price set, no_price set, age ≤ threshold.
    Returns False for replay/backtest where no live book exists.  O(1).
    """
    if orderbook is None:
        return False
    if orderbook.yes_price is None or orderbook.no_price is None:
        return False
    age = orderbook.age_s()
    # age_s() returns None when updated_at is unset (replay snapshots)
    if age is None or age > _ORDERBOOK_MAX_AGE_S:
        return False
    return True


def _spread_from_orderbook(orderbook: OrderbookSnapshot) -> float:
    """Derive spread from real top-of-book quotes.

    yes_ask = 1.0 - no_price  (the NO price implies the YES ask)
    spread  = yes_ask - yes_price
    Clamped to [0.01, 0.10] to stay in the same range as the estimator.
    """
    yes_ask = 1.0 - orderbook.no_price  # type: ignore[operator]
    spread = yes_ask - orderbook.yes_price  # type: ignore[operator]
    return max(0.01, min(0.10, spread))


# ── Spread estimation ────────────────────────────────────────────────

def estimate_spread(
    mid_price: float,
    time_to_expiry_h: float,
    volume_1m: float,
    base_spread: float = 0.03,
) -> float:
    """Estimate bid-ask spread when no orderbook is available.

    Uses price-level entropy, time-to-expiry, and recent volume as proxies.
    Returns spread in [0.01, 0.10] — clamped so it stays realistic.

    Args:
        mid_price: YES-side market price (0..1)
        time_to_expiry_h: hours until contract expiry
        volume_1m: contracts traded in the last minute
        base_spread: median observed spread (default 3 cents)
    """
    # Variance proxy: p*(1-p) peaks at 0.50 (0.25), approaches 0 at extremes.
    # Spreads widen at price extremes where LPs quote less aggressively.
    variance = mid_price * (1.0 - mid_price)
    price_factor = 1.0 + 2.0 * (0.25 - variance)

    # Time factor: spreads widen further from expiry (less certainty)
    tte_ratio = min(time_to_expiry_h / 24.0, 1.0)
    tte_factor = 1.0 + 0.3 * tte_ratio

    # Volume factor: more volume -> tighter spread
    vol_factor = 1.0 / (1.0 + volume_1m / 10.0)

    spread = base_spread * price_factor * tte_factor * vol_factor
    # Clamp to realistic Kalshi range
    return max(0.01, min(0.10, spread))


# ── Slippage ─────────────────────────────────────────────────────────

def compute_slippage(size: int, top_of_book_size: int = 10) -> float:
    """Size-dependent slippage from walking the orderbook.

    Each price level beyond top-of-book costs an extra cent.

    Args:
        size: number of contracts in this order
        top_of_book_size: estimated resting quantity at best price
    """
    if size <= top_of_book_size:
        return 0.0
    # Each additional level costs 0.01 (1 cent)
    levels_walked = math.ceil((size - top_of_book_size) / top_of_book_size)
    return 0.01 * levels_walked


# ── Fill probability ─────────────────────────────────────────────────

def compute_fill_probability(
    offset: float,
    spread: float,
    volume_1m: float,
) -> float:
    """Probability that a limit order fills during the decision interval.

    Market orders (offset == 0) always fill (return 1.0).
    Limit orders deeper inside the spread have lower fill probability.

    Args:
        offset: price improvement requested (0 = market order)
        spread: current estimated bid-ask spread
        volume_1m: contracts traded in the last minute
    """
    # Market orders always fill
    if offset <= 0.0:
        return 1.0

    # How far the limit price sits inside the spread (0 = at best, 1 = at mid)
    penetration = offset / max(spread, 0.01)

    # Exponential decay: deeper limits fill less often
    base_fill = math.exp(-2.0 * penetration)

    # Volume boost: busier markets fill limits more often
    vol_boost = min(volume_1m / 20.0, 1.0)

    fill_prob = base_fill * (0.5 + 0.5 * vol_boost)
    return max(0.05, min(1.0, fill_prob))


# ── Fee classification ───────────────────────────────────────────────

def classify_fee_rate(
    offset: float,
    is_close: bool,
    maker_rate: float = 0.0175,
    taker_rate: float = 0.07,
) -> float:
    """Return the fee rate for this trade.

    Market orders (offset == 0) and all closes pay taker fee.
    Limit orders (offset > 0) pay maker fee.
    """
    if is_close or offset <= 0.0:
        return taker_rate
    return maker_rate


# ── Top-level compute ────────────────────────────────────────────────

def compute_execution(
    direction: str,
    size: int,
    offset: float,
    mid_price: float,
    time_to_expiry_h: float,
    volume_1m: float,
    is_close: bool = False,
    base_spread: float = 0.03,
    top_of_book_size: int = 10,
    maker_rate: float = 0.0175,
    taker_rate: float = 0.07,
    orderbook: OrderbookSnapshot | None = None,
) -> ExecutionResult:
    """Compute realistic execution price, fees, and fill probability.

    Combines spread estimation, slippage, fill probability, and fee
    classification into a single call for use in environment.step().

    When *orderbook* is fresh (both sides quoted, updated within 30 s),
    real spread and resting depth replace the heuristic estimates.
    Otherwise the estimation path runs unchanged (replay / backtest).

    Args:
        direction: "yes" or "no"
        size: number of contracts
        offset: price improvement requested (0 = market order)
        mid_price: current YES-side mid-market price
        time_to_expiry_h: hours until expiry
        volume_1m: recent 1-minute volume
        is_close: True if closing an existing position
        base_spread: base spread param for estimate_spread
        top_of_book_size: resting quantity at best price
        maker_rate: maker fee rate
        taker_rate: taker fee rate
        orderbook: live top-of-book snapshot (None in replay)

    Returns:
        ExecutionResult with exec_price, fee_rate, fill_prob, spread, slippage
    """
    # ── Spread & depth: prefer live orderbook, fall back to estimation ──
    if _orderbook_is_fresh(orderbook):
        spread = _spread_from_orderbook(orderbook)  # type: ignore[arg-type]
        # Direction-aware depth: opens use opposite side, closes use same side
        if is_close:
            real_depth = orderbook.yes_size if direction == "yes" else orderbook.no_size  # type: ignore[union-attr]
        else:
            real_depth = orderbook.no_size if direction == "yes" else orderbook.yes_size  # type: ignore[union-attr]
        effective_depth = max(1, real_depth)
    else:
        spread = estimate_spread(mid_price, time_to_expiry_h, volume_1m, base_spread)
        effective_depth = top_of_book_size

    half = spread / 2.0
    slip = compute_slippage(size, effective_depth)
    fill_prob = compute_fill_probability(offset, spread, volume_1m)
    fee_rate = classify_fee_rate(offset, is_close, maker_rate, taker_rate)

    # Compute execution price based on direction and order type
    if is_close:
        # Closes always cross the spread (taker) + slippage
        if direction == "yes":
            # Selling YES hits the bid
            exec_price = mid_price - half - slip
        else:
            # Selling NO hits the ask (YES-side price goes up)
            exec_price = mid_price + half + slip
    else:
        # Opens
        if direction == "yes":
            if offset <= 0.0:
                # Market buy YES: lift the ask
                exec_price = mid_price + half + slip
            else:
                # Limit buy YES: bid below ask
                exec_price = mid_price + half - offset
        else:
            if offset <= 0.0:
                # Market buy NO: hit the bid (YES-side price goes down)
                exec_price = mid_price - half - slip
            else:
                # Limit buy NO: offer above bid
                exec_price = mid_price - half + offset

    # Clamp to valid Kalshi price range [0.01, 0.99]
    exec_price = max(0.01, min(0.99, exec_price))

    return ExecutionResult(
        exec_price=exec_price,
        fee_rate=fee_rate,
        fill_prob=fill_prob,
        spread=spread,
        slippage=slip,
    )
