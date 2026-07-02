import math
from dataclasses import dataclass


def compute_maker_fee(contracts: int, price: float, fee_rate: float) -> float:
    """Calculate maker fee using Kalshi's variance-based model.

    Formula: fee_rate * contracts * price * (1 - price), rounded up to nearest cent.
    Peaks at price=0.50, approaches zero at extremes.

    Args:
        contracts: number of contracts traded
        price: YES-side market price (0.0 to 1.0)
        fee_rate: base maker fee rate (e.g., 0.0175 for 1.75%)

    Returns:
        Fee amount rounded up to nearest cent.
    """
    # Variance-based model: variance is maximized at price=0.5
    raw = fee_rate * contracts * price * (1.0 - price)
    # Round up to nearest cent (0.01)
    return math.ceil(raw * 100) / 100


def compute_taker_fee(contracts: int, price: float, fee_rate: float) -> float:
    """Calculate taker fee — same variance-based formula as the maker fee,
    but at Kalshi's taker rate (0.07 vs 0.0175 maker). Live data shows 94.5%
    of our fills end up taker (we cross the spread), so the simulator charges
    this rate on most fills to keep sim fee drag honest.

    Args:
        contracts: number of contracts traded
        price: YES-side market price (0.0 to 1.0)
        fee_rate: base taker fee rate (e.g., 0.07 for 7%)

    Returns:
        Fee amount rounded up to nearest cent.
    """
    raw = fee_rate * contracts * price * (1.0 - price)
    return math.ceil(raw * 100) / 100


def fee_at_quote_size(contracts: int, price: float, fee_rate: float,
                      quote_size: int) -> float:
    """Fee for `contracts` executed as separate quote_size-lot orders.

    Kalshi rounds each ORDER's fee up to the next cent. The live bot quotes
    quote_size-lot orders, so a simulated fill of N contracts really executes
    as ceil(N/quote_size) separate orders, each paying its own ceil'd fee.
    At quote_size=1 and price 0.50 that's 1 cent per contract — 2.3x the
    nominal rate — which the single-ceil sim fee badly understated.

    A partial final lot is charged the full per-lot fee (pessimistic by at
    most one sub-cent rounding; keeps the formula O(1) and branch-free).
    """
    quote_size = max(quote_size, 1)
    n_lots = math.ceil(contracts / quote_size)
    per_lot = math.ceil(fee_rate * quote_size * price * (1.0 - price) * 100) / 100
    return per_lot * n_lots


@dataclass
class _Position:
    """Internal record of an open position.

    Stores position metadata needed to calculate PnL and fees on close/settle.
    """
    direction: str  # "yes" or "no"
    size: int  # number of contracts
    entry_price: float  # YES-side price at entry (for consistency)
    entry_fee: float  # fee paid at entry


class PnLTracker:
    """Tracks positions, realized PnL, and fees across all markets.

    Positions are stored with YES-side pricing internally for consistency.
    A "no" position stores the YES price at entry (NO cost is 1 - yes_price,
    but internally we use YES price for unified PnL calculation).

    Time complexity: O(1) for all operations (dictionary lookups).
    """

    def __init__(self, maker_fee_rate: float = 0.0175) -> None:
        """Initialize tracker with fee rate.

        Args:
            maker_fee_rate: base maker fee rate (default 0.0175 for 1.75%)
        """
        self._fee_rate = maker_fee_rate
        # ticker -> _Position for open positions. O(1) lookup.
        self._positions: dict[str, _Position] = {}
        # cumulative realized PnL for the day
        self._daily_pnl = 0.0

    def open_position(self, ticker: str, direction: str, size: int, price: float) -> None:
        """Record opening a new position.

        Args:
            ticker: market ticker (e.g., "KXBTC-A")
            direction: "yes" or "no"
            size: number of contracts (positive)
            price: current YES-side market price at entry
        """
        entry_fee = compute_maker_fee(size, price, self._fee_rate)
        self._positions[ticker] = _Position(
            direction=direction,
            size=size,
            entry_price=price,
            entry_fee=entry_fee,
        )

    def close_position(self, ticker: str, close_price: float = 0.0) -> float:
        """Close an open position and return realized PnL after fees.

        Args:
            ticker: market ticker
            close_price: current YES-side market price at close

        Returns:
            Net PnL after entry + exit fees. Adds to daily total.
        """
        pos = self._positions.pop(ticker, None)
        if pos is None:
            return 0.0

        # Gross PnL depends on direction
        if pos.direction == "yes":
            # Long YES: profit when YES price goes up
            gross = (close_price - pos.entry_price) * pos.size
        else:
            # Long NO (short YES): profit when YES price goes down
            gross = (pos.entry_price - close_price) * pos.size

        # Exit fee (pay when closing)
        exit_fee = compute_maker_fee(pos.size, close_price, self._fee_rate)

        # Net PnL after both entry and exit fees
        net = gross - pos.entry_fee - exit_fee
        self._daily_pnl += net
        return net

    def settle(self, ticker: str, outcome: bool) -> float:
        """Settle a position at contract expiry.

        Args:
            ticker: market ticker
            outcome: True if YES wins (settles at 1.00), False if NO wins (settles at 0.00)

        Returns:
            Net PnL after entry fee (no exit fee on settlement). Adds to daily total.
        """
        pos = self._positions.pop(ticker, None)
        if pos is None:
            return 0.0

        # Settlement price for YES side: 1.00 if YES wins, 0.00 if NO wins
        settle_price = 1.0 if outcome else 0.0

        # Gross PnL at settlement
        if pos.direction == "yes":
            gross = (settle_price - pos.entry_price) * pos.size
        else:
            gross = (pos.entry_price - settle_price) * pos.size

        # No exit fee on settlement (only entry fee)
        net = gross - pos.entry_fee
        self._daily_pnl += net
        return net

    def get_position(self, ticker: str) -> int:
        """Get signed position for a ticker.

        Args:
            ticker: market ticker

        Returns:
            +N = long YES (size N), -N = long NO (size N), 0 = flat or no position
        """
        pos = self._positions.get(ticker)
        if pos is None:
            return 0
        return pos.size if pos.direction == "yes" else -pos.size

    def get_unrealized_pnl(self, ticker: str, market_price: float) -> float:
        """Mark-to-market unrealized PnL (before fees).

        Args:
            ticker: market ticker
            market_price: current YES-side market price

        Returns:
            Unrealized PnL if position exists, 0.0 if flat.
        """
        pos = self._positions.get(ticker)
        if pos is None:
            return 0.0
        if pos.direction == "yes":
            # Long YES: gains when price rises
            return (market_price - pos.entry_price) * pos.size
        else:
            # Long NO: gains when YES price falls
            return (pos.entry_price - market_price) * pos.size

    def get_entry_price(self, ticker: str) -> float | None:
        """Return entry YES-side price for a ticker.

        Args:
            ticker: market ticker

        Returns:
            YES-side entry price, or None if no position.
        """
        pos = self._positions.get(ticker)
        return pos.entry_price if pos is not None else None

    def total_exposure(self) -> int:
        """Count of markets with non-zero positions.

        Returns:
            Number of open positions across all tickers.
        """
        return len(self._positions)

    def daily_pnl(self) -> float:
        """Cumulative realized PnL for the current day.

        Returns:
            Sum of all realized PnL from closes and settlements today.
        """
        return self._daily_pnl

    def reset_daily(self) -> None:
        """Reset daily PnL counter (call at start of new trading day)."""
        self._daily_pnl = 0.0
