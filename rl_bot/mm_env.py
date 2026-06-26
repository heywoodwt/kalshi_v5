"""Market-making environment with multi-ticker support."""

import gymnasium
from rl_bot.mm_config import MMConfig
from rl_bot.mm_metadata import MarketMetadataLoader


class MMEnv(gymnasium.Env):
    """Market-making environment with multi-ticker support."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        ticker_data: dict[str, list[dict]],
        cfg: MMConfig,
        metadata_loader: MarketMetadataLoader | None = None,
    ):
        """Initialize environment with preprocessed trade data.

        Args:
            ticker_data: output of preprocess_mm_data()
            cfg: MMConfig instance with all hyperparameters
            metadata_loader: Market metadata loader for tick validation
        """
        super().__init__()

        self._cfg = cfg
        self._ticker_data = ticker_data
        self._tickers = list(ticker_data.keys())
        self._ticker_idx = -1
        self._metadata_loader = metadata_loader
        self._current_ticker = None

    def _apply_subpenny_adjustment(self, price: float, side: str) -> float:
        """Apply subpenny adjustment for queue priority if market supports it.

        Args:
            price: Base price computed from mid + spread + skew
            side: "bid" or "ask"

        Returns:
            Adjusted price (or original if subpenny not valid)
        """
        # Check if metadata loader available
        if self._metadata_loader is None:
            return price  # No validation possible, return original

        # Check if current market supports subpenny at this price
        if not self._metadata_loader.supports_subpenny(self._current_ticker, price):
            return price  # No adjustment if tick size is 0.01

        # Apply queue-jumping adjustment
        if side == "bid":
            # Bid: add 0.001 to jump ahead (pay more)
            adjusted = price + 0.001
        else:  # side == "ask"
            # Ask: subtract 0.001 to jump ahead (sell cheaper)
            adjusted = price - 0.001

        # Ensure we stay within valid Kalshi range [0.01, 0.99]
        adjusted = max(0.01, min(0.99, adjusted))

        # Round to valid tick (0.001 precision)
        adjusted = round(adjusted, 3)

        return adjusted
