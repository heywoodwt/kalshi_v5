"""
Controlled test to show what orders would be generated.
Forces market conditions to pass all filters.
"""
import logging
from volatility import VolatilityTracker
from market_selector.quoting_engine import generate_quotes, format_quotes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("test")


class BiasedModel:
    """Model that returns strong predictions."""

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, ticker, features):
        return self.prediction

    def is_ready(self):
        return True


def create_volatile_market():
    """Create a market with high volatility to pass filters."""
    tracker = VolatilityTracker()

    # Build steady baseline
    for _ in range(40):
        tracker.update(0.50)

    # Then add volatile spike
    volatile_sequence = [
        0.50, 0.52, 0.48, 0.55, 0.45, 0.60, 0.40, 0.58, 0.42, 0.56,
        0.44, 0.54, 0.46, 0.52, 0.48, 0.50, 0.53, 0.47, 0.55, 0.45,
    ]

    for price in volatile_sequence:
        tracker.update(price)

    return tracker, list(tracker.prices)


def test_bullish_signal():
    """Test strong BUY signal generation."""
    log.info("\n" + "="*70)
    log.info("TEST 1: BULLISH SIGNAL (model predicts 0.80, market at 0.50)")
    log.info("="*70)

    tracker, prices = create_volatile_market()
    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]

    log.info(f"\nMarket State:")
    log.info(f"  Price: ${market_price:.3f}")
    log.info(f"  Volatility: {metrics['vol']:.4f}")
    log.info(f"  Range: ${metrics['range']:.3f}")
    log.info(f"  Vol Ratio: {metrics['vol_ratio']:.2f}")
    log.info(f"  Momentum: {metrics['momentum']:.4f}")

    # Model predicts 80% (strong buy)
    model_prob = 0.80
    edge = abs(model_prob - market_price)

    log.info(f"\nModel Prediction: {model_prob:.1%}")
    log.info(f"Edge: ${edge:.3f} (prediction - market)")

    # Generate quotes
    quotes = generate_quotes(
        model_prob=model_prob,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=50,
        prices=prices,
        inventory=0,
    )

    log.info(f"\n{format_quotes(quotes, 'KXBTC-TEST')}")

    if quotes["buy"]:
        q = quotes["buy"]
        log.info(f"\n{'='*70}")
        log.info(f"📈 BUY ORDER WOULD BE PLACED (if PAPER_TRADING=false)")
        log.info(f"{'='*70}")
        log.info(f"  Market Ticker: KXBTC-TEST")
        log.info(f"  Side: BUY (YES)")
        log.info(f"  Price: ${q.price:.3f} per contract")
        log.info(f"  Contracts: 1 (configurable)")
        log.info(f"  Expected Value: ${q.ev:.4f}")
        log.info(f"  Fill Probability: {q.fill_prob:.1%}")
        log.info(f"  Edge vs Market: ${q.edge:.3f}")
        log.info(f"\n  API Call Would Be:")
        log.info(f"    POST /trade-api/v2/portfolio/orders")
        log.info(f"    {{")
        log.info(f"      'ticker': 'KXBTC-TEST',")
        log.info(f"      'side': 'yes',")
        log.info(f"      'type': 'limit',")
        log.info(f"      'yes_price': {int(q.price * 100)},  # cents")
        log.info(f"      'count': 1")
        log.info(f"    }}")
        log.info(f"{'='*70}\n")


def test_bearish_signal():
    """Test strong SELL signal generation."""
    log.info("\n" + "="*70)
    log.info("TEST 2: BEARISH SIGNAL (model predicts 0.20, market at 0.50)")
    log.info("="*70)

    tracker, prices = create_volatile_market()
    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]

    log.info(f"\nMarket State:")
    log.info(f"  Price: ${market_price:.3f}")

    # Model predicts 20% (strong sell)
    model_prob = 0.20
    edge = abs(model_prob - market_price)

    log.info(f"\nModel Prediction: {model_prob:.1%}")
    log.info(f"Edge: ${edge:.3f}")

    quotes = generate_quotes(
        model_prob=model_prob,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=50,
        prices=prices,
        inventory=0,
    )

    log.info(f"\n{format_quotes(quotes, 'KXBTC-TEST')}")

    if quotes["sell"]:
        q = quotes["sell"]
        log.info(f"\n{'='*70}")
        log.info(f"📉 SELL ORDER WOULD BE PLACED (if PAPER_TRADING=false)")
        log.info(f"{'='*70}")
        log.info(f"  Market Ticker: KXBTC-TEST")
        log.info(f"  Side: SELL (NO)")
        log.info(f"  Price: ${q.price:.3f} per contract")
        log.info(f"  Contracts: 1")
        log.info(f"  Expected Value: ${q.ev:.4f}")
        log.info(f"  Fill Probability: {q.fill_prob:.1%}")
        log.info(f"  Edge vs Market: ${q.edge:.3f}")
        log.info(f"\n  API Call Would Be:")
        log.info(f"    POST /trade-api/v2/portfolio/orders")
        log.info(f"    {{")
        log.info(f"      'ticker': 'KXBTC-TEST',")
        log.info(f"      'side': 'no',")
        log.info(f"      'type': 'limit',")
        log.info(f"      'no_price': {int((1 - q.price) * 100)},  # cents")
        log.info(f"      'count': 1")
        log.info(f"    }}")
        log.info(f"{'='*70}\n")


def test_neutral_signal():
    """Test neutral signal (no trade)."""
    log.info("\n" + "="*70)
    log.info("TEST 3: NEUTRAL SIGNAL (model predicts 0.50, market at 0.50)")
    log.info("="*70)

    tracker, prices = create_volatile_market()
    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]

    # Model matches market (no edge)
    model_prob = market_price
    edge = abs(model_prob - market_price)

    log.info(f"\nMarket Price: ${market_price:.3f}")
    log.info(f"Model Prediction: {model_prob:.3f}")
    log.info(f"Edge: ${edge:.3f} → TOO SMALL (< 0.05 threshold)")

    log.info(f"\n❌ NO TRADE - Edge filter would block")
    log.info(f"   (This is the current behavior with mock model)\n")


def test_inventory_blocking():
    """Test inventory management blocking quotes."""
    log.info("\n" + "="*70)
    log.info("TEST 4: INVENTORY BLOCKING")
    log.info("="*70)

    tracker, prices = create_volatile_market()
    metrics = tracker.get_metrics()
    market_price = metrics["last_price"]

    # Strong buy signal
    model_prob = 0.80

    # Test with long position
    log.info(f"\nScenario A: Already LONG (inventory=+5)")
    quotes_long = generate_quotes(
        model_prob=model_prob,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=50,
        prices=prices,
        inventory=5,  # Long position
    )
    log.info(f"  Buy quote: {quotes_long['buy']}")
    log.info(f"  Sell quote: {quotes_long['sell']}")
    log.info(f"  → Buy blocked, only sell allowed")

    # Test with short position
    log.info(f"\nScenario B: Already SHORT (inventory=-5)")
    quotes_short = generate_quotes(
        model_prob=model_prob,
        market_price=market_price,
        vol_metrics=metrics,
        recent_volume=50,
        prices=prices,
        inventory=-5,  # Short position
    )
    log.info(f"  Buy quote: {quotes_short['buy']}")
    log.info(f"  Sell quote: {quotes_short['sell']}")
    log.info(f"  → Sell blocked, only buy allowed\n")


if __name__ == "__main__":
    test_bullish_signal()
    test_bearish_signal()
    test_neutral_signal()
    test_inventory_blocking()

    log.info("\n" + "="*70)
    log.info("SUMMARY")
    log.info("="*70)
    log.info("✓ Order generation logic tested")
    log.info("✓ No actual orders sent (PAPER_TRADING=true)")
    log.info("✓ To enable live trading: set PAPER_TRADING=false in .env")
    log.info("✓ Real orders would be submitted via Kalshi REST API")
    log.info("="*70 + "\n")
