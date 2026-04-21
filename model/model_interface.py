"""Model interface for price prediction in prediction markets.

Bridges the v5 real-time trading system with the TFT model.

Two modes:
  1. TFT mode: Loads a trained checkpoint, buffers streaming price data,
     periodically runs TFT inference, and returns cached predictions.
  2. Passthrough mode: No checkpoint found — returns market price (zero edge).
     Used during development or before a model is trained.

The predict() method always returns a probability in [0, 1].
"""

import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default checkpoint directory (relative to project root)
_DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / "tft" / "checkpoints"

# Minimum observations per ticker before TFT inference is attempted
_MIN_BUFFER_SIZE = 70

# Seconds between TFT inference runs (avoids running every tick)
_INFERENCE_COOLDOWN = 30.0

# Maximum observations to keep per ticker
_MAX_BUFFER_SIZE = 500


class ModelInterface:
    """Prediction model interface for estimating market fair value.

    On init, attempts to load a trained TFT checkpoint from model/tft/checkpoints/.
    If found, buffers incoming price data and runs periodic TFT inference.
    If not found, falls back to passthrough (returns market price).
    """

    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        # TFT predictor (None if no checkpoint or import fails)
        self._tft = None

        # Per-ticker price buffer: ticker -> deque of (timestamp, yes_price_cents)
        self._buffer: Dict[str, deque] = {}

        # Cached TFT predictions: ticker -> probability in [0, 1]
        self._predictions: Dict[str, float] = {}

        # Timestamp of last inference run
        self._last_inference: float = 0.0

        # Always ready (passthrough works without model)
        self._ready = True

        # Try to load trained TFT model
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CHECKPOINT_DIR
        self._try_load_tft(ckpt_dir)

    def _try_load_tft(self, checkpoint_dir: Path) -> None:
        """Attempt to load a trained TFT model from checkpoint directory.

        Looks for: tft_model.ckpt, tft_model.json, tft_model_dataset.pt
        If any are missing or imports fail, falls back to passthrough mode.
        """
        ckpt_path = checkpoint_dir / "tft_model"
        ckpt_file = ckpt_path.with_suffix(".ckpt")

        if not ckpt_file.exists():
            logger.info("No TFT checkpoint at %s — using passthrough mode", ckpt_file)
            return

        try:
            from model.tft.tft_model import TFTPredictor
            self._tft = TFTPredictor.load(ckpt_path)
            logger.info("TFT model loaded from %s", ckpt_path)
        except ImportError as e:
            logger.warning("TFT dependencies not installed (%s) — using passthrough mode", e)
        except Exception as e:
            logger.warning("Failed to load TFT model (%s) — using passthrough mode", e)

    def predict(self, ticker: str, features: Dict[str, Any]) -> float:
        """Predict fair value probability for a market.

        If TFT model is loaded:
          - Buffers the current price observation
          - Periodically runs TFT inference across all buffered tickers
          - Returns cached TFT prediction (converted from cents to [0, 1])

        If no TFT model:
          - Returns market_price (passthrough, zero edge)

        Args:
            ticker: Market ticker symbol
            features: Dict of features including:
                - market_price: Current market price (0.0 to 1.0)
                - vol, momentum, range, vol_ratio, etc.

        Returns:
            Fair value estimate in [0, 1]
        """
        market_price = features.get("market_price", 0.5)

        # If TFT is loaded, buffer data and run inference periodically
        if self._tft is not None:
            # Buffer this observation (convert price from [0,1] to cents [0,100])
            self._buffer_observation(ticker, market_price)

            # Check if it's time to run inference
            now = time.monotonic()
            if now - self._last_inference >= _INFERENCE_COOLDOWN:
                self._run_inference()
                self._last_inference = now

            # Return cached prediction if available
            if ticker in self._predictions:
                return self._predictions[ticker]

        # Fallback: passthrough (no predictive edge)
        return market_price

    def _buffer_observation(self, ticker: str, market_price: float) -> None:
        """Add a price observation to the per-ticker buffer.

        Args:
            ticker: Market ticker symbol
            market_price: Price in [0, 1] (converted to cents for TFT)
        """
        if ticker not in self._buffer:
            self._buffer[ticker] = deque(maxlen=_MAX_BUFFER_SIZE)

        self._buffer[ticker].append({
            "timestamp": datetime.now(timezone.utc),
            "ticker": ticker,
            "yes_price": market_price * 100.0,  # TFT expects cents (0-100)
        })

    def _run_inference(self) -> None:
        """Run TFT inference on all buffered tickers and update prediction cache.

        Builds a Polars DataFrame from buffered observations, runs feature
        engineering, and feeds through the TFT model. Predictions are converted
        from cents to [0, 1] probability and cached per ticker.
        """
        if self._tft is None:
            return

        try:
            import polars as pl
            from model.tft.feature_engineer import engineer_features

            # Build DataFrame from all buffered observations
            all_rows = []
            for ticker, obs_deque in self._buffer.items():
                # Only include tickers with enough data
                if len(obs_deque) < _MIN_BUFFER_SIZE:
                    continue
                all_rows.extend(obs_deque)

            if not all_rows:
                return

            df = pl.DataFrame(all_rows)

            # Run feature engineering (handles missing columns gracefully)
            features_df = engineer_features(df)
            if features_df.height < _MIN_BUFFER_SIZE:
                return

            # Run TFT inference
            predictions = self._tft.predict(features_df)

            # Update cache: convert cents -> probability [0, 1]
            for pred in predictions:
                # Clamp predicted price to [0, 100] cents, then normalize to [0, 1]
                prob = max(0.0, min(1.0, pred.predicted_price / 100.0))
                self._predictions[pred.ticker] = prob

            logger.info(
                "TFT inference: %d tickers, predictions updated",
                len(predictions),
            )

        except Exception as e:
            logger.warning("TFT inference failed: %s", e)

    def is_ready(self) -> bool:
        """Check if model is ready to make predictions.

        Returns:
            True if model is loaded and ready (always True — passthrough is always ready)
        """
        return self._ready