"""
Test script to simulate trading pipeline without connecting to Kalshi.
Shows what orders would be generated if PAPER_TRADING=false.
"""
import logging
from config import (
    VOL_RATIO_THRESHOLD, RANGE_THRESHOLD, MIN_EDGE, MIN_VOLUME,
    BUY_PROB_THRESHOLD, SELL_PROB_THRESHOLD, EDGE_VOL_THRESHOLD,
)
from market_selector.market_filter import is_btc_market
from volatility import VolatilityTracker
from market_selector.quoting_engine import generate_quotes, format_quotes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("test")


class MockModel:
    """Mock model that returns biased predictions to trigger trades."""

    def __init__(self, prediction_type="bullish"):
        self.prediction_type = prediction_type
        self._ready = True

    def predict(self, ticker, features):
        market_price = features.get("market_price", 0.5)

        if self.prediction_type == "bullish":
            # Strong buy signal: predict much higher than market
            return min(market_price + 0.25, 0.95)
        elif self.prediction_type == "bearish":
            # Strong sell signal: predict much lower than market
            return max(market_price - 0.25, 0.05)
        else:
            # Neutral: no edge
            return market_price

    def is_ready(self):
        return self._ready


def simulate_market_activity(ticker, num_updates=50):
    """Simulate realistic market price movements."""
    import random

    log.info(f"\n{'='*60}")
    log.info(f"Simulating market activity for {ticker}")
    log.info(f"{'='*60}")

    tracker = VolatilityTracker()
    trade_count = 0

    # Start at 0.50, simulate volatile moves
    base_price = 0.50
    prices = []

    for i in range(num_updates):
        # Add volatility: random walk with occasional spikes
        if random.random() < 0.1:  # 10% chance of spike
            move = random.uniform(-0.08, 0.08)
        else:
            move = random.uniform(-0.02, 0.02)

        base_price = max(0.01, min(0.99, base_price + move))
        prices.append(base_price)
        tracker.update(base_price)

        # Simulate trades
        if random.random() < 0.3:
            trade_count += random.randint(1, 3)

    return tracker, trade_count, prices


def test_trade_pipeline(ticker, model, tracker, trade_count, prices):
    """Test the complete trade filter and quoting pipeline."""

    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]

    log.info(f"\n--- Market Metrics ---")
    log.info(f"Market price: ${market_price:.3f}")
    log.info(f"Volatility: {metrics['vol']:.4f}")
    log.info(f"Range: {metrics['range']:.3f}")
    log.info(f"Vol ratio: {metrics['vol_ratio']:.2f}")
    log.info(f"Momentum: {metrics['momentum']:.4f}")
    log.info(f"Trade count: {trade_count}")

    # Apply filters sequentially
    log.info(f"\n--- Filter Pipeline ---")

    # Filter 1: Volatility
    if metrics["vol_ratio"] < VOL_RATIO_THRESHOLD:
        log.warning(f"❌ FAILED: Vol ratio {metrics['vol_ratio']:.2f} < {VOL_RATIO_THRESHOLD}")
        return None
    log.info(f"✓ Vol ratio {metrics['vol_ratio']:.2f} >= {VOL_RATIO_THRESHOLD}")

    if metrics["range"] < RANGE_THRESHOLD:
        log.warning(f"❌ FAILED: Range {metrics['range']:.3f} < {RANGE_THRESHOLD}")
        return None
    log.info(f"✓ Range {metrics['range']:.3f} >= {RANGE_THRESHOLD}")

    # Filter 2: Volume
    if trade_count < MIN_VOLUME:
        log.warning(f"❌ FAILED: Volume {trade_count} < {MIN_VOLUME}")
        return None
    log.info(f"✓ Volume {trade_count} >= {MIN_VOLUME}")

    # Filter 3: Model prediction
    if not model.is_ready():
        log.warning(f"❌ FAILED: Model not ready")
        return None

    features = {"market_price": market_price, **metrics}
    pred = model.predict(ticker, features)
    log.info(f"✓ Model prediction: {pred:.3f}")

    # Filter 4: Edge
    edge = abs(pred - market_price)
    if edge < MIN_EDGE:
        log.warning(f"❌ FAILED: Edge {edge:.3f} < {MIN_EDGE}")
        return None
    log.info(f"✓ Edge {edge:.3f} >= {MIN_EDGE}")

    # Filter 5: Direction
    if not (pred > BUY_PROB_THRESHOLD or pred < SELL_PROB_THRESHOLD):
        log.warning(f"❌ FAILED: Prediction {pred:.3f} not strong enough")
        return None

    direction = "BUY" if pred > BUY_PROB_THRESHOLD else "SELL"
    log.info(f"✓ Direction filter passed: {direction} signal")

    # Filter 6: Combined
    combined = edge * metrics["vol_ratio"]
    if combined < EDGE_VOL_THRESHOLD:
        log.warning(f"❌ FAILED: Edge×VolRatio {combined:.3f} < {EDGE_VOL_THRESHOLD}")
        return None
    log.info(f"✓ Combined filter {combined:.3f} >= {EDGE_VOL_THRESHOLD}")

    # Generate quotes
    log.info(f"\n--- Quote Generation ---")
    quotes = generate_quotes(
        model_prob=pred,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=trade_count,
        prices=prices,
        inventory=0,  # flat position
    )

    return quotes


def main():
    """Run trading simulation tests."""

    test_cases = [
        ("KXBTC-26APR19-100000", "bullish"),
        ("KXBTC-26APR19-100000", "bearish"),
        ("KXBTC-26APR19-100000", "neutral"),
    ]

    for ticker, prediction_type in test_cases:
        # Check market filter
        assert is_btc_market(ticker), f"Market filter failed for {ticker}"

        # Simulate market activity
        tracker, trade_count, prices = simulate_market_activity(ticker, num_updates=100)

        # Create model with bias
        model = MockModel(prediction_type=prediction_type)

        # Run pipeline
        quotes = test_trade_pipeline(ticker, model, tracker, trade_count, prices)

        # Show results
        if quotes and (quotes["buy"] or quotes["sell"]):
            log.info(f"\n{'='*60}")
            log.info(f"🎯 TRADE SIGNAL GENERATED")
            log.info(f"{'='*60}")
            log.info(f"\n{format_quotes(quotes, ticker)}")

            log.info(f"\n--- Order Details (if PAPER_TRADING=false) ---")
            for side in ("buy", "sell"):
                q = quotes.get(side)
                if q:
                    log.info(f"\n{side.upper()} ORDER:")
                    log.info(f"  Price: ${q.price:.3f}")
                    log.info(f"  Expected Value: ${q.ev:.4f}")
                    log.info(f"  Fill Probability: {q.fill_prob:.1%}")
                    log.info(f"  Edge: ${q.edge:.3f}")
                    log.info(f"  → Would submit: market={ticker}, side={side}, price={q.price}")
        else:
            log.info(f"\n❌ No trade signal (filters blocked)")

        log.info(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
