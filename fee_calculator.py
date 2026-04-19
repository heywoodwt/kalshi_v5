import math
from config import MAKER_FEE_RATE, TAKER_FEE_RATE


def maker_fee(contracts, price):
    raw = MAKER_FEE_RATE * contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100


def taker_fee(contracts, price):
    raw = TAKER_FEE_RATE * contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100
