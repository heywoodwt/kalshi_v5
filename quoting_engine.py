from dataclasses import dataclass
from fill_probability import fill_probability
from fee_calculator import maker_fee
from config import GRID_OFFSETS, SUBPENNY_IMPROVEMENT


@dataclass
class Quote:
    price: float
    side: str
    ev: float
    fill_prob: float
    edge: float


def compute_ev(candidate_price, model_prob, side, market_price, momentum, recent_volume, prices):
    fp = fill_probability(candidate_price, market_price, momentum, recent_volume, prices)
    if side == "buy":
        edge = model_prob - candidate_price
    else:
        edge = candidate_price - (1 - model_prob)
    fee = maker_fee(1, candidate_price)
    ev = fp * edge - fee
    return Quote(price=candidate_price, side=side, ev=ev, fill_prob=fp, edge=edge)


def generate_quotes(model_prob, market_price, vol_metrics, recent_volume, prices, inventory=0):
    momentum = vol_metrics.get("momentum", 0.0)
    candidates = {"buy": [], "sell": []}

    for offset in GRID_OFFSETS:
        buy_price = round(market_price - offset, 3)
        if 0 < buy_price < 1:
            q = compute_ev(buy_price, model_prob, "buy", market_price, momentum, recent_volume, prices)
            candidates["buy"].append(q)

        sell_price = round(market_price + offset, 3)
        if 0 < sell_price < 1:
            q = compute_ev(sell_price, model_prob, "sell", market_price, momentum, recent_volume, prices)
            candidates["sell"].append(q)

    result = {"buy": None, "sell": None}

    if inventory <= 0:
        buys = [q for q in candidates["buy"] if q.ev > 0]
        if buys:
            best = max(buys, key=lambda q: q.ev)
            best.price = round(best.price + SUBPENNY_IMPROVEMENT, 3)
            result["buy"] = best

    if inventory >= 0:
        sells = [q for q in candidates["sell"] if q.ev > 0]
        if sells:
            best = max(sells, key=lambda q: q.ev)
            best.price = round(best.price - SUBPENNY_IMPROVEMENT, 3)
            result["sell"] = best

    return result


def format_quotes(quotes, ticker):
    parts = [f"[QUOTE] {ticker}"]
    for side in ("buy", "sell"):
        q = quotes.get(side)
        if q:
            parts.append(
                f"  {side.upper()}: {q.price:.3f} "
                f"(EV={q.ev:.4f}, fill={q.fill_prob:.3f}, edge={q.edge:.3f})"
            )
    return "\n".join(parts) if len(parts) > 1 else f"[QUOTE] {ticker}: no quotes"
