"""
Fee calculation for Kalshi binary options market trades.

Kalshi uses a variance-based fee model where fees are proportional to
price * (1 - price), which peaks at price = 0.5 (maximum uncertainty)
and approaches zero at extremes (0 or 1).

Fees are rounded up to the nearest cent per Kalshi API requirements.
"""
import math
from config import MAKER_FEE_RATE, TAKER_FEE_RATE


def maker_fee(contracts: int, price: float) -> float:
    """
    Calculate maker fee for a limit order that provides liquidity.

    Market makers are rewarded with lower fees (vs takers) for providing liquidity.
    Formula: fee_rate × contracts × price × (1 - price)
    Rounded up to nearest cent (Kalshi requirement).

    Args:
        contracts: Number of contracts to trade
        price: Contract price between 0.0 and 1.0

    Returns:
        Fee amount in dollars, rounded up to nearest cent

    Example:
        >>> maker_fee(1, 0.50)  # 1 contract at $0.50
        0.01  # For MAKER_FEE_RATE = 0.0175
    """
    # Variance-based fee: peaks at price=0.5, minimum at extremes
    raw = MAKER_FEE_RATE * contracts * price * (1 - price)

    # Round up to nearest cent per Kalshi API rules
    return math.ceil(raw * 100) / 100


def taker_fee(contracts: int, price: float) -> float:
    """
    Calculate taker fee for a market order that removes liquidity.

    Taker fees are higher than maker fees since they consume existing liquidity.
    Formula: fee_rate × contracts × price × (1 - price)
    Rounded up to nearest cent (Kalshi requirement).

    Args:
        contracts: Number of contracts to trade
        price: Contract price between 0.0 and 1.0

    Returns:
        Fee amount in dollars, rounded up to nearest cent

    Example:
        >>> taker_fee(1, 0.50)  # 1 contract at $0.50
        0.02  # For TAKER_FEE_RATE = 0.07
    """
    # Variance-based fee: peaks at price=0.5, minimum at extremes
    raw = TAKER_FEE_RATE * contracts * price * (1 - price)

    # Round up to nearest cent per Kalshi API rules
    return math.ceil(raw * 100) / 100
