import math
from config import FILL_D0, FILL_WEIGHTS


def distance_score(order_price, market_price):
    dist = abs(order_price - market_price)
    return math.exp(-dist / FILL_D0)


def momentum_score(order_price, market_price, momentum):
    direction = 1 if order_price > market_price else -1
    aligned = direction * momentum
    return 1 / (1 + math.exp(-aligned * 100))


def volume_score(recent_volume, scale=20):
    return min(recent_volume / scale, 1.0)


def proximity_score(prices, order_price, threshold=0.02):
    if not prices:
        return 0.0
    within = sum(1 for p in prices if abs(p - order_price) <= threshold)
    return within / len(prices)


def fill_probability(order_price, market_price, momentum, recent_volume, prices):
    w = FILL_WEIGHTS
    d = distance_score(order_price, market_price)
    m = momentum_score(order_price, market_price, momentum)
    v = volume_score(recent_volume)
    p = proximity_score(prices, order_price)
    return w[0] * d + w[1] * m + w[2] * v + w[3] * p
