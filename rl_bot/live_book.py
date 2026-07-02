"""Canonical live orderbook for Kalshi + pure quote-safety helpers.

Why this exists: Kalshi delivers orderbook data in three shapes with
conflicting orderings —
  - REST snapshots: yes/no arrays sorted ascending, best price LAST
  - WS snapshots: same arrays, often one side per message
  - WS deltas: single (side, price, delta_qty) adjustments
Reading a fixed index (e.g. levels[0]) is therefore wrong at least some of
the time. LiveBook stores each side as a dict keyed by price, so ordering
never matters: the best level is always computed with max().

Everything is expressed in YES terms:
  - "yes" side = bids to buy YES  -> our best bid = max(yes prices)
  - "no" side  = bids to buy NO   -> a NO bid at p sells YES at 1-p,
                                     so our best ask = 1 - max(no prices)

Rust translation note: dict[float, float] maps to a BTreeMap<u32, f64>
keyed by price in tenths-of-a-cent; max() becomes last_key_value().
"""

import math

# Prices are dollars with at most 3 decimals (subpenny tick = 0.001).
# Round keys to 4 decimals so float noise can't split one level into two.
_KEY_DECIMALS = 4

# Snapshot payloads name each side differently across REST and WS versions.
_YES_KEYS = ("yes_dollars_fp", "yes_dollars", "yes")
_NO_KEYS = ("no_dollars_fp", "no_dollars", "no")


class LiveBook:
    """Orderbook state for one market, safe against level-ordering bugs."""

    def __init__(self):
        self._yes: dict[float, float] = {}  # YES bid price -> contracts
        self._no: dict[float, float] = {}   # NO bid price -> contracts

    # --- ingestion ------------------------------------------------------------

    @staticmethod
    def _parse_levels(levels: list) -> dict[float, float]:
        """Convert [[price, qty], ...] (any order, str or float) to a dict."""
        out: dict[float, float] = {}
        for lv in levels:
            price = round(float(lv[0]), _KEY_DECIMALS)
            qty = float(lv[1])
            if qty > 0:
                out[price] = qty
        return out

    def load_snapshot(self, payload: dict) -> None:
        """Load a REST or WS snapshot. Replaces only the sides present,
        so Kalshi's one-side-per-message WS snapshots merge correctly."""
        for key in _YES_KEYS:
            if key in payload:
                self._yes = self._parse_levels(payload[key])
                break
        for key in _NO_KEYS:
            if key in payload:
                self._no = self._parse_levels(payload[key])
                break

    def apply_delta(self, side: str, price: float, delta_qty: float) -> None:
        """Apply a WS orderbook_delta: adjust one level's quantity.
        Levels at qty <= 0 are removed."""
        levels = self._yes if side == "yes" else self._no
        key = round(float(price), _KEY_DECIMALS)
        new_qty = levels.get(key, 0.0) + float(delta_qty)
        if new_qty > 0:
            levels[key] = new_qty
        else:
            levels.pop(key, None)

    # --- top of book (always computed, never index-dependent) -------------------

    def best_bid(self) -> float | None:
        """Highest YES bid, or None if the bid side is empty."""
        return max(self._yes) if self._yes else None

    def best_ask(self) -> float | None:
        """Lowest YES ask = 1 - highest NO bid, or None if the NO side is empty."""
        return round(1.0 - max(self._no), _KEY_DECIMALS) if self._no else None

    def is_valid(self) -> bool:
        """True when both sides exist and the book is not crossed."""
        bid, ask = self.best_bid(), self.best_ask()
        return bid is not None and ask is not None and ask > bid

    def mid(self) -> float | None:
        """Midpoint of a valid book, else None."""
        if not self.is_valid():
            return None
        return (self.best_bid() + self.best_ask()) / 2.0

    def spread(self) -> float | None:
        """Bid-ask spread of a valid book, else None."""
        if not self.is_valid():
            return None
        return self.best_ask() - self.best_bid()

    # --- depth (for observation features) ---------------------------------------

    def bid_levels(self, n: int) -> list[tuple[float, float]]:
        """Top n YES bid levels as (price, qty), best first."""
        prices = sorted(self._yes, reverse=True)[:n]
        return [(p, self._yes[p]) for p in prices]

    def ask_levels(self, n: int) -> list[tuple[float, float]]:
        """Top n YES ask levels as (price, qty), best (lowest ask) first.
        Prices are converted from NO bids; sizes are the NO-side sizes."""
        prices = sorted(self._no, reverse=True)[:n]  # highest NO bid = best ask
        return [(round(1.0 - p, _KEY_DECIMALS), self._no[p]) for p in prices]


# --- pure quote-safety helpers (unit-testable, no I/O) ---------------------------


def sample_mid(hist: list, last_sample_ts: float, now_ts: float, mid: float,
               window_s: float = 60.0, max_len: int = 20) -> float:
    """Maintain a mid-price history sampled on the training cadence.

    Training features (momentum, velocity, realized vol, and the 0.05 vol
    filter threshold) were all computed on 60-second windows. Live book ticks
    arrive every ~1 second; appending each one made those features run ~60x
    too fast and the vol filter meaningless. Instead: append one entry per
    window, and refresh the last entry in place between window boundaries
    (mirroring how a window's VWAP evolves while the window is open).

    Mutates `hist` in place. Returns the (possibly advanced) sample timestamp
    the caller must store for the next call.
    """
    if not hist or now_ts - last_sample_ts >= window_s:
        hist.append(mid)
        if len(hist) > max_len:
            del hist[: len(hist) - max_len]  # keep the newest max_len entries
        return now_ts
    hist[-1] = mid  # same window still open — update, don't append
    return last_sample_ts


def trade_window_features(trades: list, now_s: float,
                          window_s: float = 60.0) -> tuple[float, float]:
    """Compute obs [9] (trade volume) and [16] (flow imbalance) from real
    trade prints, exactly as training does per 60s window.

    `trades` is a list of (epoch_s, contracts, taker_side) tuples, mutated in
    place to drop prints older than the window (keeps memory bounded without
    a separate pruning pass).

    Training semantics replicated:
      volume_1m = number of PRINTS / 50, capped at 1.0   (mm_env n_trades/50)
      flow      = (buy_vol - sell_vol) / total contracts, where buy means
                  taker_side == "no" (taker bought NO -> the maker bought YES)
                  — matching mm_env's side_int encoding, not intuition.
    """
    cutoff = now_s - window_s
    # Prints arrive time-ordered; drop stale entries from the front
    while trades and trades[0][0] < cutoff:
        trades.pop(0)
    if not trades:
        return 0.0, 0.0
    buy_vol = sum(c for _, c, side in trades if side == "no")
    sell_vol = sum(c for _, c, side in trades if side == "yes")
    volume_1m = min(len(trades) / 50.0, 1.0)
    flow = (buy_vol - sell_vol) / max(buy_vol + sell_vol, 1)
    return volume_1m, flow


def clamp_quotes(bid: float, ask: float, best_bid: float, best_ask: float,
                 tick: float) -> tuple[float, float] | None:
    """Force quotes to be passive: bid stays under the market ask, ask stays
    over the market bid. A quote that crosses the touch would execute as
    taker (paying the spread + 7% taker fee) — the opposite of market making.

    Returns (bid, ask) or None when the clamped quotes would cross each other
    (market too tight to quote both sides passively).
    """
    # Round before comparing — float subtraction noise (0.47-0.01=0.45999...)
    # would otherwise let a bid==ask pair slip through the crossing check.
    bid = round(min(bid, best_ask - tick), _KEY_DECIMALS)
    ask = round(max(ask, best_bid + tick), _KEY_DECIMALS)
    if ask <= bid:
        return None
    return bid, ask


def quote_edge_ok(bid: float, ask: float, fee_rate: float, size: int,
                  min_edge: float) -> bool:
    """True when the quoted spread beats the round-trip maker fees plus a
    minimum edge, per contract.

    Kalshi rounds each fill's fee UP to the next cent (see reward.compute_maker_fee),
    so at size=1 a round trip near 0.50 costs $0.02 regardless of the nominal
    1.75% rate. Quoting a spread that fees consume guarantees a loss on every
    completed round trip — refuse to quote instead.
    """
    # Per-fill ceil'd fees for each leg at this order size
    fee_buy = math.ceil(fee_rate * size * bid * (1.0 - bid) * 100) / 100
    fee_sell = math.ceil(fee_rate * size * ask * (1.0 - ask) * 100) / 100
    # Spread captured on a full round trip of `size` contracts vs fees + required
    # edge. Round to kill float noise (0.51-0.49 != 0.02 exactly in binary).
    captured = round((ask - bid) * size, 6)
    return captured > round(fee_buy + fee_sell + min_edge * size, 6)
