"""
Market filtering and tracking for targeted market types.

Filters incoming market data to focus only on configured market prefix
(e.g., BTC-related markets). Also tracks discovered markets to avoid
redundant processing.
"""
from config import MARKET_PREFIX

# Set of markets we've already discovered and processed
# Prevents logging spam when same market appears repeatedly in feed
_known_markets: set[str] = set()


def is_btc_market(ticker: str) -> bool:
    """
    Check if a market ticker matches our target market type.

    Uses prefix matching to identify relevant markets (e.g., all BTC markets).
    Configured via MARKET_PREFIX in config.py.

    Args:
        ticker: Market ticker symbol (e.g., "KXBTC-26APR19-100000")

    Returns:
        True if ticker starts with MARKET_PREFIX, False otherwise
    """
    return ticker.startswith(MARKET_PREFIX)


def register_market(ticker: str) -> bool:
    """
    Register a newly discovered market to avoid duplicate processing.

    Maintains a set of known markets so we can log new discoveries
    without spamming for every ticker update.

    Args:
        ticker: Market ticker symbol to register

    Returns:
        True if this is a new market (just registered)
        False if market was already known
    """
    if ticker in _known_markets:
        # Already seen this market
        return False

    # New market - add to known set
    _known_markets.add(ticker)
    return True
