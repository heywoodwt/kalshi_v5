"""
Quote generation engine for market making in prediction markets.

Generates optimal buy/sell quotes by:
1. Evaluating candidate prices on a grid around market price
2. Calculating expected value (EV) for each candidate
3. Selecting quotes with positive EV that maximize profit
4. Applying sub-penny price improvement for better queue position

EV formula: (fill_probability × edge) - fees
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from market_selector.fill_probability import fill_probability
from market_selector.fee_calculator import maker_fee
from config import GRID_OFFSETS, SUBPENNY_IMPROVEMENT

# Default number of contracts per order
DEFAULT_CONTRACTS = 1


@dataclass
class Quote:
    """
    Represents a potential limit order quote with its metrics.

    Attributes:
        price: Limit order price (0.0 to 1.0)
        side: "buy" or "sell"
        ev: Expected value in dollars
        fill_prob: Estimated probability of fill (0.0 to 1.0)
        edge: Theoretical profit if filled (before fees)
    """
    price: float
    side: str
    ev: float
    fill_prob: float
    edge: float


def compute_ev(
    candidate_price: float,
    model_prob: float,
    side: str,
    market_price: float,
    momentum: float,
    recent_volume: int,
    prices: List[float]
) -> Quote:
    """
    Calculate expected value (EV) for a potential limit order.

    EV = (fill_probability × edge) - fees

    Where edge is the theoretical profit if the order fills:
    - For buy orders: edge = model_prob - price (we think it's worth more)
    - For sell orders: edge = price - (1 - model_prob) (we think it's worth less)

    Args:
        candidate_price: Proposed limit order price
        model_prob: Our model's fair value estimate
        side: "buy" or "sell"
        market_price: Current market mid price
        momentum: Recent price momentum
        recent_volume: Recent trading volume
        prices: Historical prices for proximity analysis

    Returns:
        Quote object with price, EV, fill probability, and edge
    """
    # Estimate probability this order will fill
    fp = fill_probability(candidate_price, market_price, momentum, recent_volume, prices)

    # Calculate theoretical edge (profit if filled, before fees)
    if side == "buy":
        # Buy at candidate_price, model says fair value is model_prob
        edge = model_prob - candidate_price
    else:
        # Sell at candidate_price, model says fair value is model_prob
        # (selling "no" at price P is like selling "yes" at 1-P)
        edge = candidate_price - (1 - model_prob)

    # Calculate fees for this order
    fee = maker_fee(DEFAULT_CONTRACTS, candidate_price)

    # Expected value = expected profit - fees
    ev = fp * edge - fee

    return Quote(price=candidate_price, side=side, ev=ev, fill_prob=fp, edge=edge)


def generate_quotes(
    model_prob: float,
    market_price: float,
    vol_metrics: Dict[str, float],
    recent_volume: int,
    prices: List[float],
    inventory: int = 0
) -> Dict[str, Optional[Quote]]:
    """
    Generate optimal buy and sell quotes for a market.

    Strategy:
    1. Generate candidate quotes at price grid offsets from market
    2. Calculate EV for each candidate
    3. Select highest-EV quote on each side (if EV > 0)
    4. Apply sub-penny price improvement for queue priority
    5. Respect inventory limits (don't add to existing position)

    Args:
        model_prob: Our model's fair value estimate (0.0 to 1.0)
        market_price: Current market mid price
        vol_metrics: Volatility metrics including momentum
        recent_volume: Recent trading volume
        prices: Historical prices
        inventory: Current position (positive = long, negative = short)

    Returns:
        Dict with "buy" and "sell" keys, values are Quote or None
    """
    # Extract momentum from volatility metrics
    momentum = vol_metrics.get("momentum", 0.0)

    # Storage for candidate quotes on each side
    candidates: Dict[str, List[Quote]] = {"buy": [], "sell": []}

    # Generate candidate quotes at grid offsets from market price
    for offset in GRID_OFFSETS:
        # Buy quotes: below market price
        buy_price = round(market_price - offset, 3)
        if 0 < buy_price < 1:  # Ensure valid price range
            q = compute_ev(buy_price, model_prob, "buy", market_price, momentum, recent_volume, prices)
            candidates["buy"].append(q)

        # Sell quotes: above market price
        sell_price = round(market_price + offset, 3)
        if 0 < sell_price < 1:  # Ensure valid price range
            q = compute_ev(sell_price, model_prob, "sell", market_price, momentum, recent_volume, prices)
            candidates["sell"].append(q)

    result: Dict[str, Optional[Quote]] = {"buy": None, "sell": None}

    # Select best buy quote (if not already long)
    if inventory <= 0:
        # Filter to positive-EV quotes only
        buys = [q for q in candidates["buy"] if q.ev > 0]
        if buys:
            # Select quote with highest expected value
            best = max(buys, key=lambda q: q.ev)

            # Apply sub-penny improvement to jump ahead in queue
            # (Kalshi uses price-time priority, so better price = better position)
            best.price = round(best.price + SUBPENNY_IMPROVEMENT, 3)
            result["buy"] = best

    # Select best sell quote (if not already short)
    if inventory >= 0:
        # Filter to positive-EV quotes only
        sells = [q for q in candidates["sell"] if q.ev > 0]
        if sells:
            # Select quote with highest expected value
            best = max(sells, key=lambda q: q.ev)

            # Apply sub-penny improvement (lower sell price = better queue position)
            best.price = round(best.price - SUBPENNY_IMPROVEMENT, 3)
            result["sell"] = best

    return result


def format_quotes(quotes: Dict[str, Optional[Quote]], ticker: str) -> str:
    """
    Format quotes for human-readable logging.

    Args:
        quotes: Dict with "buy" and "sell" keys
        ticker: Market ticker symbol

    Returns:
        Formatted multi-line string showing quote details
    """
    parts = [f"[QUOTE] {ticker}"]

    for side in ("buy", "sell"):
        q = quotes.get(side)
        if q:
            # Display quote with all key metrics
            parts.append(
                f"  {side.upper()}: {q.price:.3f} "
                f"(EV={q.ev:.4f}, fill={q.fill_prob:.3f}, edge={q.edge:.3f})"
            )

    # Return formatted string, or indicate no quotes
    return "\n".join(parts) if len(parts) > 1 else f"[QUOTE] {ticker}: no quotes"
