from config import MARKET_PREFIX

_known_markets = set()


def is_btc_market(ticker):
    return ticker.startswith(MARKET_PREFIX)


def register_market(ticker):
    if ticker in _known_markets:
        return False
    _known_markets.add(ticker)
    return True
