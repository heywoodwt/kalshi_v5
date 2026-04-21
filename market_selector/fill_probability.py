"""
Fill probability estimation for limit orders in prediction markets.

Estimates the likelihood that a limit order will fill based on:
- Distance from market price (closer = more likely to fill)
- Price momentum alignment (momentum toward our price = more likely)
- Recent trading volume (higher volume = more likely)
- Historical price proximity (price has been near our level = more likely)

Uses weighted combination of these factors to produce probability in [0, 1].
"""
import math
from typing import List, Tuple
from config import FILL_D0, FILL_WEIGHTS

# Constants for score calculations
VOLUME_SCALE = 20          # Volume normalization factor
PROXIMITY_THRESHOLD = 0.02  # Price proximity threshold (2 cents)
MOMENTUM_SCALE = 100        # Momentum sigmoid steepness


def distance_score(order_price: float, market_price: float) -> float:
    """
    Calculate fill probability based on distance from market price.

    Uses exponential decay: closer orders are much more likely to fill.
    Returns 1.0 when order_price == market_price, decays to ~0 as distance grows.

    Args:
        order_price: Our limit order price
        market_price: Current market mid price

    Returns:
        Score in [0, 1], higher = more likely to fill
    """
    dist = abs(order_price - market_price)
    # Exponential decay with characteristic distance FILL_D0
    return math.exp(-dist / FILL_D0)


def momentum_score(order_price: float, market_price: float, momentum: float) -> float:
    """
    Calculate fill probability based on price momentum alignment.

    If price is moving toward our limit order level, it's more likely to fill.
    - Positive alignment (momentum toward our order): score > 0.5
    - Negative alignment (momentum away): score < 0.5

    Args:
        order_price: Our limit order price
        market_price: Current market mid price
        momentum: Recent net price change (positive = upward movement)

    Returns:
        Score in [0, 1], higher = momentum favors our order filling
    """
    # Determine if our order is above or below market
    direction = 1 if order_price > market_price else -1

    # Positive aligned means momentum is pushing price toward our order
    aligned = direction * momentum

    # Sigmoid function: maps momentum to probability
    # Large positive aligned → score near 1.0
    # Large negative aligned → score near 0.0
    return 1 / (1 + math.exp(-aligned * MOMENTUM_SCALE))


def volume_score(recent_volume: int, scale: int = VOLUME_SCALE) -> float:
    """
    Calculate fill probability based on recent trading volume.

    Higher volume means more trading activity and higher fill probability.
    Capped at 1.0 to avoid over-weighting this factor.

    Args:
        recent_volume: Number of recent trades
        scale: Volume level that gives score of 1.0

    Returns:
        Score in [0, 1], higher = more active market
    """
    # Linear scale, capped at 1.0
    return min(recent_volume / scale, 1.0)


def proximity_score(prices: List[float], order_price: float, threshold: float = PROXIMITY_THRESHOLD) -> float:
    """
    Calculate fill probability based on historical price proximity.

    If market price has been near our order level recently, it's more likely
    to return there and fill our order.

    Args:
        prices: Recent historical prices
        order_price: Our limit order price
        threshold: Price distance threshold to consider "nearby"

    Returns:
        Score in [0, 1]: fraction of recent prices within threshold
    """
    if not prices:
        return 0.0

    # Count prices that were near our order level
    within = sum(1 for p in prices if abs(p - order_price) <= threshold)

    # Return as fraction
    return within / len(prices)


def fill_probability(
    order_price: float,
    market_price: float,
    momentum: float,
    recent_volume: int,
    prices: List[float]
) -> float:
    """
    Estimate overall fill probability using weighted combination of factors.

    Combines four independent signals:
    1. Distance from market (exponential decay)
    2. Momentum alignment (sigmoid)
    3. Recent volume (linear)
    4. Historical proximity (frequency)

    Weights are configured in FILL_WEIGHTS tuple.

    Args:
        order_price: Our limit order price
        market_price: Current market mid price
        momentum: Recent net price change
        recent_volume: Number of recent trades
        prices: Recent historical prices

    Returns:
        Estimated fill probability in [0, 1]
    """
    w = FILL_WEIGHTS

    # Calculate individual component scores
    d = distance_score(order_price, market_price)
    m = momentum_score(order_price, market_price, momentum)
    v = volume_score(recent_volume)
    p = proximity_score(prices, order_price)

    # Weighted combination
    return w[0] * d + w[1] * m + w[2] * v + w[3] * p
