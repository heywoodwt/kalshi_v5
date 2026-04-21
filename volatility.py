from collections import deque
from typing import Dict
from config import VOL_WINDOW

# Constants for metric calculation windows
RECENT_VOL_WINDOW = 10  # Window size for recent volatility calculation
MOMENTUM_WINDOW = 5     # Window size for momentum calculation


class VolatilityTracker:
    """
    Tracks rolling price volatility and market dynamics for a single market.

    Uses a fixed-size window to calculate metrics like volatility, momentum,
    and price range. Optimized for single-pass calculation to minimize CPU usage
    in real-time streaming context.
    """

    def __init__(self, window: int = VOL_WINDOW) -> None:
        """
        Initialize volatility tracker with fixed window size.

        Args:
            window: Number of price updates to track (default from config)
        """
        self.prices = deque(maxlen=window)

    def update(self, price: float) -> None:
        """
        Add new price observation to the tracker.

        Args:
            price: Latest market price (0.0 to 1.0 for binary markets)
        """
        self.prices.append(price)

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate all volatility metrics in a single pass through price data.

        Optimized to avoid multiple iterations over the same data. Computes:
        - vol: Average absolute price change
        - range: Max price - min price in window
        - vol_ratio: Recent volatility / average volatility (spike detector)
        - vol_up: Average upward movement
        - vol_down: Average downward movement
        - momentum: Sum of recent price changes (trend indicator)
        - last_price: Most recent price

        Returns:
            Dictionary of metric name -> value
        """
        if len(self.prices) < 2:
            # Not enough data for meaningful metrics
            return {
                "vol": 0.0,
                "range": 0.0,
                "vol_ratio": 0.0,
                "vol_up": 0.0,
                "vol_down": 0.0,
                "momentum": 0.0,
                "last_price": self.prices[-1] if self.prices else 0.0,
            }

        # Single-pass calculation of all metrics
        # Track: min, max, diffs, up_sum, down_sum, recent_vol_sum, momentum_sum
        prices_iter = iter(self.prices)
        prev = next(prices_iter)

        min_price = prev
        max_price = prev

        total_abs_diff = 0.0
        up_sum = 0.0
        up_count = 0
        down_sum = 0.0
        down_count = 0

        diffs = []  # Store for recent window calculations

        for curr in prices_iter:
            diff = curr - prev
            diffs.append(diff)

            # Update min/max for range
            if curr < min_price:
                min_price = curr
            if curr > max_price:
                max_price = curr

            # Update volatility components
            abs_diff = abs(diff)
            total_abs_diff += abs_diff

            if diff > 0:
                up_sum += diff
                up_count += 1
            elif diff < 0:
                down_sum += abs_diff
                down_count += 1

            prev = curr

        # Calculate aggregate metrics
        n_diffs = len(diffs)
        last_price = prev

        # Average volatility across all diffs
        avg_vol = total_abs_diff / n_diffs

        # Price range
        price_range = max_price - min_price

        # Directional volatility
        vol_up = up_sum / up_count if up_count > 0 else 0.0
        vol_down = down_sum / down_count if down_count > 0 else 0.0

        # Recent volatility (last N diffs) for spike detection
        recent_window = min(RECENT_VOL_WINDOW, n_diffs)
        recent_vol = sum(abs(d) for d in diffs[-recent_window:]) / recent_window

        # Avoid division by zero
        vol_ratio = recent_vol / avg_vol if avg_vol > 1e-9 else 0.0

        # Momentum (net price movement in recent window)
        momentum_window = min(MOMENTUM_WINDOW, n_diffs)
        momentum = sum(diffs[-momentum_window:])

        return {
            "vol": avg_vol,
            "range": price_range,
            "vol_ratio": vol_ratio,
            "vol_up": vol_up,
            "vol_down": vol_down,
            "momentum": momentum,
            "last_price": last_price,
        }
