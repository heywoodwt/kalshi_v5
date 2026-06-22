import logging
from dataclasses import dataclass, field

import numpy as np

from model.hp_dfm_rte.orderbook import OrderbookSnapshot
from order_management.volatility import VolatilityTracker
from rl_bot.btc_data import BTCDataPoller
from rl_bot.config import (
    ACTION_CLOSE_NO,
    ACTION_CLOSE_YES,
    ACTION_HOLD,
    RLConfig,
    decode_action,
)
from rl_bot.reward import PnLTracker
from rl_bot.state_builder import build_action_mask, build_state

log = logging.getLogger(__name__)

# Minimum price updates before a market is considered active
_MIN_OBSERVATIONS = 2


@dataclass
class _MarketState:
    """Internal per-market tracking data."""
    vol_tracker: VolatilityTracker = field(default_factory=VolatilityTracker)
    orderbook: OrderbookSnapshot = field(default_factory=lambda: OrderbookSnapshot(ticker=""))
    trade_count: int = 0
    time_to_expiry_h: float = 0.0
    last_price: float = 0.0
    observation_count: int = 0


class TradingEnv:
    """Trading environment that wraps Kalshi market state for the RL agent.

    Receives market data updates via on_ticker/on_trade/on_orderbook callbacks,
    tracks positions via PnLTracker, and executes actions (paper or live).
    """

    def __init__(self, cfg: RLConfig, btc_poller: BTCDataPoller) -> None:
        self._cfg = cfg
        self._btc = btc_poller
        self._pnl = PnLTracker(maker_fee_rate=cfg.maker_fee_rate)
        # Per-market state, lazily initialized
        self._markets: dict[str, _MarketState] = {}

    def _get_market(self, ticker: str) -> _MarketState:
        """Get or create market state for a ticker."""
        if ticker not in self._markets:
            self._markets[ticker] = _MarketState(
                orderbook=OrderbookSnapshot(ticker=ticker)
            )
        return self._markets[ticker]

    # -- Data update callbacks --

    def on_ticker(self, ticker: str, price: float, time_to_expiry_h: float) -> None:
        """Update market price and time-to-expiry from ticker message."""
        ms = self._get_market(ticker)
        ms.vol_tracker.update(price)
        ms.last_price = price
        ms.time_to_expiry_h = time_to_expiry_h
        ms.observation_count += 1

    def on_trade(self, ticker: str, price: float, count: int) -> None:
        """Update trade count and price from trade message."""
        ms = self._get_market(ticker)
        ms.vol_tracker.update(price)
        ms.trade_count += count
        ms.last_price = price
        ms.observation_count += 1

    def on_orderbook(self, ticker: str, snapshot: OrderbookSnapshot) -> None:
        """Update orderbook snapshot."""
        ms = self._get_market(ticker)
        ms.orderbook = snapshot

    # -- State queries --

    def get_active_markets(self) -> list[str]:
        """Return tickers with enough observations for meaningful state."""
        return [
            ticker for ticker, ms in self._markets.items()
            if ms.observation_count >= _MIN_OBSERVATIONS
        ]

    def get_state(self, ticker: str) -> np.ndarray:
        """Build current state vector for a market."""
        ms = self._get_market(ticker)
        vol_metrics = ms.vol_tracker.get_metrics()
        return build_state(
            ticker=ticker,
            vol_metrics=vol_metrics,
            orderbook=ms.orderbook,
            btc_poller=self._btc,
            pnl_tracker=self._pnl,
            market_price=ms.last_price,
            time_to_expiry_h=ms.time_to_expiry_h,
            trade_count=ms.trade_count,
            cfg=self._cfg,
        )

    def get_mask(self, ticker: str) -> np.ndarray:
        """Build action mask for a market."""
        return build_action_mask(ticker, self._pnl, self._cfg)

    # -- Action execution --

    def step(
        self, ticker: str, action_id: int
    ) -> tuple[np.ndarray, float, bool]:
        """Execute an action and return (next_state, reward, done).

        Args:
            ticker: market to act on
            action_id: integer action (0-20)

        Returns:
            next_state: updated state vector after action
            reward: realized PnL (0 for opens and holds)
            done: True if market is settled/removed
        """
        ms = self._get_market(ticker)
        reward = 0.0
        done = False

        decoded = decode_action(action_id)

        if decoded == "hold":
            pass  # No action

        elif decoded == "close_yes" or decoded == "close_no":
            # Close the position at current market price
            reward = self._pnl.close_position(ticker, close_price=ms.last_price)

        else:
            # Buy action: (direction, size, offset)
            direction, size, offset = decoded
            # Calculate execution price with offset
            if direction == "yes":
                # Buying YES: pay ask price + offset gives us a better (lower) price
                exec_price = max(0.01, ms.last_price - offset)
            else:
                # Buying NO: the YES-side price we record is market + offset
                exec_price = min(0.99, ms.last_price + offset)

            # Check if we already have a position in opposite direction
            current_pos = self._pnl.get_position(ticker)
            if (direction == "yes" and current_pos < 0) or (direction == "no" and current_pos > 0):
                # Close existing position first, then open new one
                reward = self._pnl.close_position(ticker, close_price=ms.last_price)

            # Open new position (or add to existing)
            self._pnl.open_position(ticker, direction, size, exec_price)

        # Build next state
        next_state = self.get_state(ticker)
        return next_state, reward, done

    def settle_market(self, ticker: str, outcome: bool) -> float:
        """Settle a market at expiry.

        Args:
            ticker: market that expired
            outcome: True if YES wins, False if NO wins

        Returns:
            Realized PnL from settlement
        """
        pnl = self._pnl.settle(ticker, outcome)
        # Clean up market state
        self._markets.pop(ticker, None)
        return pnl

    def is_circuit_breaker_active(self) -> bool:
        """True if daily realized losses exceed the configured limit."""
        return self._pnl.daily_pnl() <= -self._cfg.max_daily_loss

    @property
    def pnl_tracker(self) -> PnLTracker:
        """Expose PnL tracker for external logging."""
        return self._pnl
