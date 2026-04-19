class ModelInterface:
    def __init__(self):
        self._ready = True

    def predict(self, ticker, features):
        return features.get("market_price", 0.5)

    def is_ready(self):
        return self._ready
