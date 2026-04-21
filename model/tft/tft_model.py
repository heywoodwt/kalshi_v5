"""TFT model wrapper: train, predict, save/load.

Wraps pytorch-forecasting TemporalFusionTransformer with a high-level API
for Kalshi BTC prediction markets. Self-contained — no external pipeline deps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as ptl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from .fees import round_trip_fee
from .feature_engineer import engineer_features
from .tft_dataset import build_dataset

logger = logging.getLogger(__name__)


@dataclass
class TFTPrediction:
    """Per-ticker prediction from the TFT model."""
    ticker: str
    predicted_price: float       # Median predicted yes_price (cents)
    lower_bound: float           # 10th percentile
    upper_bound: float           # 90th percentile
    uncertainty: float           # upper - lower
    current_price: float         # Latest observed yes_price

    @property
    def deviation(self) -> float:
        """Signed deviation: predicted - current (cents)."""
        return self.predicted_price - self.current_price

    @property
    def confidence(self) -> float:
        """|deviation| / uncertainty — higher means more actionable."""
        if self.uncertainty <= 0:
            return 0.0
        return abs(self.deviation) / self.uncertainty

    @property
    def direction(self) -> str:
        """Trade direction based on deviation sign."""
        return "BUY_YES" if self.deviation > 0 else "BUY_NO"

    def net_ev(self, contracts: int = 1, maker: bool = True) -> float:
        """Expected value after fees (cents)."""
        edge = abs(self.deviation)
        entry = self.current_price if self.deviation > 0 else 100.0 - self.current_price
        exit_price = entry + edge
        entry_int = max(1, min(99, int(round(entry))))
        exit_int = max(1, min(99, int(round(exit_price))))
        fees = round_trip_fee(entry_int, exit_int, contracts=contracts, maker=maker)
        return edge - fees

    def edge_fee_ratio(self, contracts: int = 1, maker: bool = True) -> float:
        """Edge / fees — should be > 2.0-3.0 to trade."""
        edge = abs(self.deviation)
        entry = self.current_price if self.deviation > 0 else 100.0 - self.current_price
        exit_price = entry + edge
        entry_int = max(1, min(99, int(round(entry))))
        exit_int = max(1, min(99, int(round(exit_price))))
        fees = round_trip_fee(entry_int, exit_int, contracts=contracts, maker=maker)
        if fees <= 0:
            return float("inf")
        return edge / fees


class TFTPredictor:
    """High-level TFT model for Kalshi BTC prediction markets.

    Wraps pytorch-forecasting's TemporalFusionTransformer with methods for
    training, inference, and checkpoint management.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        attention_head_size: int = 2,
        dropout: float = 0.1,
        hidden_continuous_size: int = 16,
        learning_rate: float = 1e-3,
        encoder_length: int = 60,
        prediction_length: int = 10,
    ) -> None:
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.learning_rate = learning_rate
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length

        self._model: TemporalFusionTransformer | None = None
        self._trainer: ptl.Trainer | None = None
        self._training_dataset: TimeSeriesDataSet | None = None

    @property
    def model(self) -> TemporalFusionTransformer:
        if self._model is None:
            raise RuntimeError("Model not trained or loaded. Call train() or load() first.")
        return self._model

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        training_dataset: TimeSeriesDataSet,
        max_epochs: int = 50,
        gpus: int = 0,
        gradient_clip_val: float = 0.1,
        patience: int = 5,
        use_lr_finder: bool = True,
    ) -> dict[str, float]:
        """Train the TFT model.

        Returns:
            Dictionary of training metrics.
        """
        self._training_dataset = training_dataset

        # Build TFT model from dataset
        self._model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=10,
            reduce_on_plateau_patience=3,
        )

        logger.info(
            "TFT model created: %d parameters",
            sum(p.numel() for p in self._model.parameters()),
        )

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", verbose=True
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Determine accelerator
        if gpus > 0 and torch.cuda.is_available():
            accelerator = "gpu"
            devices = gpus
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = 1

        self._trainer = ptl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=gradient_clip_val,
            callbacks=[early_stop, lr_monitor],
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # Optional LR finder
        if use_lr_finder:
            try:
                lr_finder = self._trainer.tuner.lr_find(
                    self._model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    min_lr=1e-6,
                    max_lr=1e-1,
                )
                suggested_lr = lr_finder.suggestion()
                if suggested_lr is not None:
                    self._model.hparams.learning_rate = suggested_lr
                    logger.info("LR finder suggested: %.2e", suggested_lr)
            except Exception:
                logger.warning("LR finder failed, using default LR")

        # Train
        self._trainer.fit(
            self._model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Collect metrics
        best_score = float(early_stop.best_score) if early_stop.best_score else float("nan")
        metrics = {
            "best_val_loss": best_score,
            "epochs_trained": self._trainer.current_epoch,
            "n_parameters": sum(p.numel() for p in self._model.parameters()),
        }

        logger.info("Training complete: %s", metrics)
        return metrics

    def predict(
        self,
        df: pl.DataFrame,
        return_raw: bool = False,
    ) -> list[TFTPrediction]:
        """Run inference on new data.

        Args:
            df: Feature-engineered DataFrame (output of engineer_features).
            return_raw: If True, also return raw prediction tensor.

        Returns:
            List of TFTPrediction per ticker.
        """
        model = self.model
        if self._training_dataset is None:
            raise RuntimeError("No training dataset available for prediction context.")

        # Build prediction dataset from training dataset parameters
        pdf = df.to_pandas()
        pdf["time_idx"] = pdf["time_idx"].astype(int)
        pdf["ticker"] = pdf["ticker"].astype(str)

        pred_dataset = TimeSeriesDataSet.from_dataset(
            self._training_dataset, pdf, predict=True, stop_randomization=True
        )
        pred_dataloader = pred_dataset.to_dataloader(
            train=False, batch_size=128, num_workers=0
        )

        # Get predictions (quantile forecasts)
        preds = model.predict(pred_dataloader, mode="raw")
        # preds.prediction shape: (n_samples, prediction_length, n_quantiles)
        # quantiles: [0.1, 0.5, 0.9]
        pred_tensor = preds.prediction

        predictions: list[TFTPrediction] = []

        # Get decoded index for mapping predictions back to tickers
        decoded_idx = pred_dataset.decoded_index

        for i in range(min(len(decoded_idx), pred_tensor.shape[0])):
            ticker_name = decoded_idx.iloc[i]["ticker"]
            # Take the last prediction step, per quantile
            q10 = float(pred_tensor[i, -1, 0])
            q50 = float(pred_tensor[i, -1, 1])
            q90 = float(pred_tensor[i, -1, 2])

            # Clamp to valid range
            q10 = max(0.0, min(100.0, q10))
            q50 = max(0.0, min(100.0, q50))
            q90 = max(0.0, min(100.0, q90))

            # Get current price
            ticker_data = df.filter(pl.col("ticker") == ticker_name)
            current_price = float(ticker_data["yes_price"][-1]) if ticker_data.height > 0 else q50

            predictions.append(TFTPrediction(
                ticker=ticker_name,
                predicted_price=q50,
                lower_bound=q10,
                upper_bound=q90,
                uncertainty=q90 - q10,
                current_price=current_price,
            ))

        # Deduplicate: keep latest prediction per ticker
        seen = {}
        for pred in predictions:
            seen[pred.ticker] = pred
        predictions = list(seen.values())

        logger.info("Generated %d predictions", len(predictions))
        return predictions

    @staticmethod
    def compute_position_size(
        pred: TFTPrediction,
        capital: float = 100.0,
        risk_pct: float = 0.02,
        max_contracts: int = 10,
    ) -> int:
        """Size position proportional to confidence, capped at max_contracts."""
        if pred.confidence <= 0:
            return 1
        raw_size = int(pred.confidence * (capital * risk_pct) / max(abs(pred.deviation), 1.0))
        return max(1, min(max_contracts, raw_size))

    def save(self, path: str | Path) -> None:
        """Save model checkpoint, hyperparameters, and training data sample.

        Saves three files:
            {path}.ckpt       — model weights
            {path}.json        — hyperparameters
            {path}_sample.parquet — small training data sample for dataset reconstruction
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = path.with_suffix(".ckpt")
        self._trainer.save_checkpoint(str(model_path))

        # Save hyperparameters
        hparams = {
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "dropout": self.dropout,
            "hidden_continuous_size": self.hidden_continuous_size,
            "learning_rate": self.learning_rate,
            "encoder_length": self.encoder_length,
            "prediction_length": self.prediction_length,
        }
        hparams_path = path.with_suffix(".json")
        hparams_path.write_text(json.dumps(hparams, indent=2))

        # Save training dataset (needed for TimeSeriesDataSet.from_dataset on load)
        if self._training_dataset is not None:
            dataset_path = Path(str(path) + "_dataset.pt")
            torch.save(self._training_dataset, str(dataset_path))
            logger.info("Saved training dataset to %s", dataset_path)

        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> TFTPredictor:
        """Load model from checkpoint with saved training dataset.

        Expects three files at the path:
            {path}.ckpt       — model weights
            {path}.json        — hyperparameters
            {path}_dataset.pt  — training dataset (TimeSeriesDataSet)

        Returns:
            TFTPredictor with loaded weights, ready for predict().
        """
        path = Path(path)

        # Load hyperparameters
        hparams_path = path.with_suffix(".json")
        if hparams_path.exists():
            hparams = json.loads(hparams_path.read_text())
        else:
            hparams = {}

        predictor = cls(**hparams)

        # Load training dataset (needed for TimeSeriesDataSet.from_dataset during predict)
        dataset_path = Path(str(path) + "_dataset.pt")
        if dataset_path.exists():
            predictor._training_dataset = torch.load(str(dataset_path), weights_only=False)
            logger.info("Loaded training dataset from %s", dataset_path)
        else:
            logger.warning("No training dataset found at %s — predict() will fail", dataset_path)

        # Load model from checkpoint
        model_path = path.with_suffix(".ckpt")
        predictor._model = TemporalFusionTransformer.load_from_checkpoint(
            str(model_path)
        )

        logger.info("Model loaded from %s", path)
        return predictor