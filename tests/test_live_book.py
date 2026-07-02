"""Tests for rl_bot/live_book.py — canonical live orderbook + quote safety helpers.

Kalshi conventions covered:
- REST /markets/{t}/orderbook returns yes/no arrays sorted ASCENDING (best price LAST)
- WS orderbook_snapshot sends yes/no sides (sometimes one side per message)
- WS orderbook_delta sends (side, price_dollars, delta_fp) adjustments
- "yes" array = bids for YES; "no" array = bids for NO; YES ask = 1 - best NO bid
"""

import pytest

from rl_bot.live_book import LiveBook, clamp_quotes, quote_edge_ok


# --- fixtures: realistic Kalshi payloads ------------------------------------

# REST format: ascending price, best level LAST (matches data_collector/collect.py)
REST_BOOK = {
    "yes_dollars": [["0.01", "500"], ["0.40", "120"], ["0.45", "30"]],   # best YES bid = 0.45
    "no_dollars": [["0.02", "800"], ["0.50", "200"], ["0.53", "60"]],    # best NO bid = 0.53 -> YES ask = 0.47
}

# WS snapshot format: fp keys, one side per message is common
WS_YES_SNAPSHOT = {"yes_dollars_fp": [["0.10", "50"], ["0.45", "30"]]}
WS_NO_SNAPSHOT = {"no_dollars_fp": [["0.20", "90"], ["0.53", "60"]]}


def make_book() -> LiveBook:
    book = LiveBook()
    book.load_snapshot(REST_BOOK)
    return book


# --- parsing: best levels must come from the BEST end, regardless of order ---

def test_rest_snapshot_best_bid_is_highest_yes_price():
    book = make_book()
    assert book.best_bid() == pytest.approx(0.45)


def test_rest_snapshot_best_ask_is_one_minus_highest_no_price():
    # YES ask = 1 - best NO bid = 1 - 0.53 = 0.47 (NOT 1 - 0.02 = 0.98)
    book = make_book()
    assert book.best_ask() == pytest.approx(0.47)


def test_snapshot_order_does_not_matter():
    # Same book with arrays reversed (best first) must parse identically
    reversed_book = {
        "yes_dollars": list(reversed(REST_BOOK["yes_dollars"])),
        "no_dollars": list(reversed(REST_BOOK["no_dollars"])),
    }
    book = LiveBook()
    book.load_snapshot(reversed_book)
    assert book.best_bid() == pytest.approx(0.45)
    assert book.best_ask() == pytest.approx(0.47)


def test_plain_yes_no_keys_also_parse():
    book = LiveBook()
    book.load_snapshot({"yes": [["0.30", "10"]], "no": [["0.60", "20"]]})
    assert book.best_bid() == pytest.approx(0.30)
    assert book.best_ask() == pytest.approx(0.40)


def test_one_sided_ws_snapshots_merge():
    # Kalshi WS sends YES and NO snapshots as separate messages; the second
    # message must not wipe the first side.
    book = LiveBook()
    book.load_snapshot(WS_YES_SNAPSHOT)
    book.load_snapshot(WS_NO_SNAPSHOT)
    assert book.best_bid() == pytest.approx(0.45)
    assert book.best_ask() == pytest.approx(0.47)


def test_snapshot_replaces_stale_side():
    # A fresh snapshot for a side replaces that side entirely (stale levels gone)
    book = make_book()
    book.load_snapshot({"yes_dollars": [["0.30", "10"]]})
    assert book.best_bid() == pytest.approx(0.30)
    assert book.best_ask() == pytest.approx(0.47)  # NO side untouched


# --- deltas ------------------------------------------------------------------

def test_delta_adds_new_level():
    book = make_book()
    book.apply_delta("yes", 0.46, 25.0)
    assert book.best_bid() == pytest.approx(0.46)


def test_delta_updates_existing_level_quantity():
    book = make_book()
    book.apply_delta("yes", 0.45, 10.0)  # 30 + 10 = 40
    assert book.bid_levels(1) == [(pytest.approx(0.45), pytest.approx(40.0))]


def test_delta_removes_level_at_zero_or_below():
    book = make_book()
    book.apply_delta("yes", 0.45, -30.0)  # 30 - 30 = 0 -> level gone
    assert book.best_bid() == pytest.approx(0.40)


def test_delta_on_no_side_moves_ask():
    book = make_book()
    book.apply_delta("no", 0.55, 40.0)  # new best NO bid -> YES ask = 0.45
    assert book.best_ask() == pytest.approx(0.45)


# --- depth and validity --------------------------------------------------------

def test_bid_levels_sorted_best_first_with_sizes():
    book = make_book()
    levels = book.bid_levels(3)
    assert [p for p, _ in levels] == [pytest.approx(0.45), pytest.approx(0.40), pytest.approx(0.01)]
    assert levels[0][1] == pytest.approx(30.0)


def test_ask_levels_sorted_best_first_in_yes_terms():
    # YES asks = 1 - NO prices: [0.47, 0.50, 0.98], sizes from NO side
    book = make_book()
    levels = book.ask_levels(3)
    assert [p for p, _ in levels] == [pytest.approx(0.47), pytest.approx(0.50), pytest.approx(0.98)]
    assert levels[0][1] == pytest.approx(60.0)


def test_empty_or_one_sided_book_is_invalid():
    book = LiveBook()
    assert not book.is_valid()
    assert book.best_bid() is None
    book.load_snapshot(WS_YES_SNAPSHOT)
    assert not book.is_valid()  # still no ask side


def test_crossed_book_is_invalid():
    book = LiveBook()
    # best bid 0.60, best NO bid 0.50 -> ask 0.50 <= bid -> crossed
    book.load_snapshot({"yes": [["0.60", "10"]], "no": [["0.50", "10"]]})
    assert not book.is_valid()


def test_valid_book_mid_and_spread():
    book = make_book()
    assert book.is_valid()
    assert book.mid() == pytest.approx(0.46)
    assert book.spread() == pytest.approx(0.02)


# --- clamp_quotes: quotes must never cross the touch (post-only safety) -------

def test_clamp_bid_that_crosses_ask():
    # Model wants bid 0.48 but market ask is 0.47 -> clamp to 0.46 (one tick under)
    bid, ask = clamp_quotes(0.48, 0.55, best_bid=0.45, best_ask=0.47, tick=0.01)
    assert bid == pytest.approx(0.46)
    assert ask == pytest.approx(0.55)


def test_clamp_ask_that_crosses_bid():
    bid, ask = clamp_quotes(0.30, 0.44, best_bid=0.45, best_ask=0.47, tick=0.01)
    assert bid == pytest.approx(0.30)
    assert ask == pytest.approx(0.46)


def test_non_crossing_quotes_unchanged():
    bid, ask = clamp_quotes(0.40, 0.52, best_bid=0.45, best_ask=0.47, tick=0.01)
    assert bid == pytest.approx(0.40)
    assert ask == pytest.approx(0.52)


def test_one_tick_market_joins_the_touch_instead_of_crossing():
    # Tightest possible market: quotes clamp to exactly the current touch
    bid, ask = clamp_quotes(0.50, 0.50, best_bid=0.49, best_ask=0.50, tick=0.01)
    assert bid == pytest.approx(0.49)
    assert ask == pytest.approx(0.50)


def test_clamp_returns_none_for_inverted_input():
    # Garbage input (bid >= ask) must not come back as a crossed quote pair
    assert clamp_quotes(0.60, 0.40, best_bid=0.45, best_ask=0.47, tick=0.01) is None


# --- quote_edge_ok: don't quote when ceil'd fees eat the spread ----------------

def test_two_cent_spread_at_midprice_fails_fee_gate_at_size_one():
    # 1-lot at ~0.50: maker fee ceils to $0.01/side -> $0.02 round trip >= $0.02 spread
    assert not quote_edge_ok(bid=0.49, ask=0.51, fee_rate=0.0175, size=1, min_edge=0.0)


def test_five_cent_spread_at_midprice_passes_fee_gate():
    assert quote_edge_ok(bid=0.48, ask=0.53, fee_rate=0.0175, size=1, min_edge=0.0)


def test_min_edge_raises_the_bar():
    # 3c spread clears 2c fees but not 2c fees + 2c min edge
    assert quote_edge_ok(bid=0.49, ask=0.52, fee_rate=0.0175, size=1, min_edge=0.0)
    assert not quote_edge_ok(bid=0.49, ask=0.52, fee_rate=0.0175, size=1, min_edge=0.02)


def test_extreme_prices_have_lower_fees_so_tighter_spreads_pass():
    # At 0.07 the fee is 0.0175*0.07*0.93 ~= $0.0011 -> ceil $0.01/side, still 1c each.
    # Round trip $0.02 -> 3c spread passes even at extreme prices.
    assert quote_edge_ok(bid=0.06, ask=0.09, fee_rate=0.0175, size=1, min_edge=0.0)


# --- fill reconciliation: fees and realized PnL booked from exchange fills -----

def _make_trader():
    """LiveTrader in paper mode — no API calls in __init__ or _process_fill."""
    from rl_bot.live_trader_v2 import LiveTrader
    return LiveTrader(paper_mode=True)


def test_round_trip_fill_books_fees_and_realized_pnl():
    trader = _make_trader()
    # Maker buy 1 YES @ 0.40: fee ceil(0.0175*0.40*0.60*100)/100 = $0.01
    trader._process_fill({
        "trade_id": "f1", "market_ticker": "KXADP-26JUL-T0",
        "action": "buy", "side": "yes", "yes_price_dollars": 0.40,
        "count": 1, "is_taker": False,
    })
    assert trader.state.positions["KXADP-26JUL-T0"] == 1
    assert trader.state.daily_pnl == pytest.approx(-0.01)

    # Maker sell 1 YES @ 0.45 flattens the position: +0.05 gross, -0.01 fee.
    # Regression: the entry price is popped when flat — realized PnL must use
    # the PRE-fill entry (0.40), not default to the fill price (=> 0).
    trader._process_fill({
        "trade_id": "f2", "market_ticker": "KXADP-26JUL-T0",
        "action": "sell", "side": "yes", "yes_price_dollars": 0.45,
        "count": 1, "is_taker": False,
    })
    assert trader.state.positions["KXADP-26JUL-T0"] == 0
    assert trader.state.daily_pnl == pytest.approx(0.05 - 0.02)
    assert trader.state.wins == 1


def test_taker_fill_charged_at_taker_rate():
    trader = _make_trader()
    # Taker buy 1 YES @ 0.50: fee ceil(0.07*0.25*100)/100 = $0.02 (4x maker rate)
    trader._process_fill({
        "trade_id": "t1", "market_ticker": "KXBTCD-26JUL01-T50000",
        "action": "buy", "side": "yes", "yes_price_dollars": 0.50,
        "count": 1, "is_taker": True,
    })
    assert trader.state.daily_pnl == pytest.approx(-0.02)


def test_duplicate_fills_are_ignored():
    trader = _make_trader()
    fill = {
        "trade_id": "dup", "market_ticker": "KXADP-26JUL-T0",
        "action": "buy", "side": "yes", "yes_price_dollars": 0.40,
        "count": 1, "is_taker": False,
    }
    trader._process_fill(fill)
    trader._process_fill(fill)  # overlapping min_ts windows resend fills
    assert trader.state.positions["KXADP-26JUL-T0"] == 1


def test_no_side_fill_converts_to_yes_equivalent():
    trader = _make_trader()
    # Buying NO @ 0.55 = shorting YES @ 0.45
    trader._process_fill({
        "trade_id": "n1", "market_ticker": "KXADP-26JUL-T0",
        "action": "buy", "side": "no", "no_price_dollars": 0.55,
        "count": 1, "is_taker": False,
    })
    assert trader.state.positions["KXADP-26JUL-T0"] == -1
    assert trader.state.entry_prices["KXADP-26JUL-T0"] == pytest.approx(0.45)


# --- sample_mid: 60s-cadence mid history matching training window size --------

def test_sample_mid_first_call_appends():
    from rl_bot.live_book import sample_mid
    hist = []
    ts = sample_mid(hist, last_sample_ts=0.0, now_ts=1000.0, mid=0.45)
    assert hist == [0.45]
    assert ts == 1000.0


def test_sample_mid_within_window_updates_last_entry_not_append():
    # Book ticks arrive every ~1s live but training windows are 60s; updating
    # in place keeps momentum/velocity/vol features on the training timescale
    from rl_bot.live_book import sample_mid
    hist = [0.45]
    ts = sample_mid(hist, last_sample_ts=1000.0, now_ts=1030.0, mid=0.46)
    assert hist == [0.46]      # same window, value refreshed
    assert ts == 1000.0        # sample clock unchanged


def test_sample_mid_after_window_appends_new_entry():
    from rl_bot.live_book import sample_mid
    hist = [0.45]
    ts = sample_mid(hist, last_sample_ts=1000.0, now_ts=1060.0, mid=0.47)
    assert hist == [0.45, 0.47]
    assert ts == 1060.0


def test_sample_mid_trims_to_max_len():
    from rl_bot.live_book import sample_mid
    hist = [float(i) for i in range(20)]  # already at cap
    sample_mid(hist, last_sample_ts=1000.0, now_ts=1060.0, mid=99.0, max_len=20)
    assert len(hist) == 20
    assert hist[-1] == 99.0
    assert hist[0] == 1.0  # oldest entry dropped


# --- taker/maker fill accounting for the Phase 2 deployment gate ---------------

def test_process_fill_counts_taker_and_maker_separately():
    trader = _make_trader()
    trader._process_fill({
        "trade_id": "g1", "market_ticker": "KXADP-26JUL-T0",
        "action": "buy", "side": "yes", "yes_price_dollars": 0.40,
        "count": 1, "is_taker": False,
    })
    trader._process_fill({
        "trade_id": "g2", "market_ticker": "KXADP-26JUL-T0",
        "action": "sell", "side": "yes", "yes_price_dollars": 0.45,
        "count": 1, "is_taker": True,
    })
    assert trader.state.maker_fills == 1
    assert trader.state.taker_fills == 1
    assert trader.state.fees_paid == pytest.approx(0.01 + 0.02)  # maker 1c + taker 2c


# --- trade_window_features: real trade prints for obs [9] and [16] -------------

def test_trade_window_features_counts_prints_and_flow():
    from rl_bot.live_book import trade_window_features
    # (epoch_s, contracts, taker_side) — mirrors training: [9] counts PRINTS/50,
    # [16] is contract-weighted with buy = taker_side "no" (taker bought NO ->
    # maker bought YES), matching mm_env.py's side_int encoding exactly
    trades = [
        (100.0, 5, "no"),    # 5 contracts buy-side
        (130.0, 3, "yes"),   # 3 contracts sell-side
        (150.0, 2, "no"),    # 2 contracts buy-side
    ]
    volume_1m, flow = trade_window_features(trades, now_s=155.0, window_s=60.0)
    assert volume_1m == pytest.approx(3 / 50.0)          # 3 prints
    assert flow == pytest.approx((7 - 3) / 10.0)          # (buy - sell) / total


def test_trade_window_features_prunes_old_trades_in_place():
    from rl_bot.live_book import trade_window_features
    trades = [(10.0, 5, "no"), (100.0, 1, "yes")]
    volume_1m, flow = trade_window_features(trades, now_s=120.0, window_s=60.0)
    assert trades == [(100.0, 1, "yes")]                  # 10.0 pruned (>60s old)
    assert volume_1m == pytest.approx(1 / 50.0)
    assert flow == pytest.approx(-1.0)                    # all sell-side


def test_trade_window_features_empty_returns_zeros():
    from rl_bot.live_book import trade_window_features
    assert trade_window_features([], now_s=100.0) == (0.0, 0.0)


def test_fractional_fill_below_one_contract_is_ignored():
    # Kalshi reports fractional fills (count_fp 0.5); int() floored them to
    # size 0, which still bumped fee/taker counters while changing nothing.
    # Sub-1 fills are ignored entirely; position sync reconciles any drift.
    trader = _make_trader()
    trader._process_fill({
        "trade_id": "frac1", "market_ticker": "KXBTCD-26JUL02-T61099",
        "action": "buy", "side": "yes", "yes_price_dollars": 0.05,
        "count_fp": "0.5", "is_taker": False,
    })
    assert trader.state.positions["KXBTCD-26JUL02-T61099"] == 0
    assert trader.state.maker_fills == 0
    assert trader.state.fees_paid == 0.0


def test_position_value_scoped_to_active_tickers():
    # Legacy positions from prior deployments must not count against the
    # bot's position-value risk cap (117 frozen legacy positions at the 0.50
    # fallback blocked all quoting against a $40 limit)
    trader = _make_trader()
    trader.state.positions["LEGACY-27-JUNK"] = -2      # not quoted by this bot
    trader.state.positions["KXBTCD-26JUL02-T60000"] = 1
    trader.state.entry_prices["KXBTCD-26JUL02-T60000"] = 0.40
    active = {"KXBTCD-26JUL02-T60000"}
    assert trader.state.position_value(active) == pytest.approx(0.40)
    # Unscoped still counts everything (legacy at the 0.50 fallback)
    assert trader.state.position_value() == pytest.approx(0.40 + 2 * 0.50)
