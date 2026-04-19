from collections import deque
from config import VOL_WINDOW


class VolatilityTracker:
    def __init__(self, window=VOL_WINDOW):
        self.prices = deque(maxlen=window)

    def update(self, price):
        self.prices.append(price)

    def get_metrics(self):
        if len(self.prices) < 2:
            return {
                "vol": 0.0,
                "range": 0.0,
                "vol_ratio": 0.0,
                "vol_up": 0.0,
                "vol_down": 0.0,
                "momentum": 0.0,
                "last_price": self.prices[-1] if self.prices else 0.0,
            }

        prices = list(self.prices)
        diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        vol = sum(abs(d) for d in diffs) / len(diffs)
        price_range = max(prices) - min(prices)

        up_diffs = [d for d in diffs if d > 0]
        down_diffs = [d for d in diffs if d < 0]
        vol_up = sum(up_diffs) / len(up_diffs) if up_diffs else 0.0
        vol_down = sum(abs(d) for d in down_diffs) / len(down_diffs) if down_diffs else 0.0

        avg_vol = vol if vol > 0 else 1e-9
        recent = diffs[-min(10, len(diffs)):]
        recent_vol = sum(abs(d) for d in recent) / len(recent) if recent else 0.0
        vol_ratio = recent_vol / avg_vol

        momentum = sum(diffs[-min(5, len(diffs)):])

        return {
            "vol": vol,
            "range": price_range,
            "vol_ratio": vol_ratio,
            "vol_up": vol_up,
            "vol_down": vol_down,
            "momentum": momentum,
            "last_price": prices[-1],
        }
