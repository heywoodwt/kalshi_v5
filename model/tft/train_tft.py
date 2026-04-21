"""CLI entry point for TFT model training.

Usage:
    # Train with synthetic data (no API calls):
    python -m model.tft.train_tft --synthetic

    # Train from saved parquet file:
    python -m model.tft.train_tft --data-file ./data/kalshi_trades.parquet

    # Full training with exchange data enrichment:
    python -m model.tft.train_tft --data-file ./data/kalshi_trades.parquet --enrich
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TFT model for Kalshi BTC prediction markets"
    )

    # Data options
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for testing (no API calls)"
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Path to Kalshi trade data (parquet with: timestamp, ticker, yes_price, volume)"
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Fetch spot/funding/OI from exchanges to enrich training data"
    )

    # Model hyperparameters
    parser.add_argument("--encoder-length", type=int, default=60)
    parser.add_argument("--prediction-length", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    # Training options
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--no-lr-finder", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="./model/tft/checkpoints")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load or generate data ---
    logger.info("=" * 60)
    logger.info("TFT Training Pipeline")
    logger.info("=" * 60)

    if args.data_file:
        logger.info("Loading data from %s", args.data_file)
        raw_data = pl.read_parquet(args.data_file)

        # Enrich with exchange data if requested
        if args.enrich:
            logger.info("Enriching with exchange data (spot, funding, OI)...")
            import asyncio
            from .data_fetcher import build_training_data
            raw_data = asyncio.run(build_training_data(raw_data))

    elif args.synthetic:
        logger.info("Generating synthetic data for testing")
        from .data_fetcher import generate_synthetic_data
        raw_data = generate_synthetic_data(n_tickers=5, n_steps=500)

    else:
        logger.error("Must specify --data-file or --synthetic")
        sys.exit(1)

    logger.info("Raw data: %d rows, %d columns", raw_data.height, raw_data.width)

    if raw_data.height == 0:
        logger.error("No data available. Exiting.")
        sys.exit(1)

    # Save raw data
    raw_data.write_parquet(output_dir / "raw_data.parquet")

    # --- Step 2: Feature engineering ---
    logger.info("Engineering features...")
    from .feature_engineer import engineer_features

    features_df = engineer_features(
        raw_data,
        encoder_length=args.encoder_length,
        prediction_length=args.prediction_length,
    )
    logger.info("Features: %d rows, %d columns", features_df.height, features_df.width)

    if features_df.height < args.encoder_length + args.prediction_length:
        logger.error(
            "Insufficient data after feature engineering. Need at least %d rows.",
            args.encoder_length + args.prediction_length,
        )
        sys.exit(1)

    # Save features
    features_df.write_parquet(output_dir / "features.parquet")

    # --- Step 3: Build dataset and dataloaders ---
    logger.info("Building dataset and dataloaders...")
    from .tft_dataset import create_dataloaders

    train_dl, val_dl, training_ds = create_dataloaders(
        features_df,
        train_frac=0.8,
        encoder_length=args.encoder_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
    )

    # --- Step 4: Train model ---
    logger.info("Training TFT model...")
    from .tft_model import TFTPredictor

    predictor = TFTPredictor(
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        encoder_length=args.encoder_length,
        prediction_length=args.prediction_length,
    )

    metrics = predictor.train(
        train_dl, val_dl, training_ds,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        patience=args.patience,
        use_lr_finder=not args.no_lr_finder,
    )

    logger.info("Training metrics: %s", metrics)

    # --- Step 5: Save checkpoint ---
    predictor.save(output_dir / "tft_model")
    logger.info("Model saved to %s", output_dir / "tft_model")

    # --- Step 6: Run inference on validation data ---
    logger.info("Running inference on validation data...")
    predictions = predictor.predict(features_df)
    logger.info("Generated %d predictions", len(predictions))

    for pred in predictions[:5]:
        logger.info(
            "  %s: current=%.1f, predicted=%.1f [%.1f, %.1f], confidence=%.2f",
            pred.ticker, pred.current_price, pred.predicted_price,
            pred.lower_bound, pred.upper_bound, pred.confidence,
        )

    logger.info("=" * 60)
    logger.info("TFT training pipeline complete!")
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()