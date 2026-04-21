"""TimeSeriesDataSet preparation for pytorch-forecasting TFT.

Converts Polars feature DataFrames into TimeSeriesDataSet objects
with proper train/val splits (no future leakage).
"""

from __future__ import annotations

import logging

import pandas as pd
import polars as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from torch.utils.data import DataLoader

from .feature_engineer import (
    STATIC_CATEGORICALS,
    TARGET,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_UNKNOWN_REALS,
)

logger = logging.getLogger(__name__)


def build_dataset(
    df: pl.DataFrame,
    encoder_length: int = 60,
    prediction_length: int = 10,
    target: str = TARGET,
    min_encoder_length: int | None = None,
    min_prediction_length: int | None = None,
    batch_size: int = 64,
) -> TimeSeriesDataSet:
    """Build a TimeSeriesDataSet from a Polars feature DataFrame.

    Args:
        df: Feature-engineered DataFrame with time_idx, ticker, and all
            feature columns (output of feature_engineer.engineer_features).
        encoder_length: Number of historical time steps the model sees.
        prediction_length: Number of future time steps to predict.
        target: Target column name.
        min_encoder_length: Minimum encoder length (defaults to encoder_length // 2).
        min_prediction_length: Minimum prediction length (defaults to 1).
        batch_size: Batch size for dataloaders.

    Returns:
        TimeSeriesDataSet configured for TFT training.
    """
    if min_encoder_length is None:
        min_encoder_length = encoder_length // 2
    if min_prediction_length is None:
        min_prediction_length = 1

    # Convert to pandas (pytorch-forecasting requirement)
    pdf = df.to_pandas()

    # Ensure time_idx is integer
    pdf["time_idx"] = pdf["time_idx"].astype(int)

    # Ensure ticker is string/category
    pdf["ticker"] = pdf["ticker"].astype(str)

    # Filter feature columns to only those present in the DataFrame
    available_unknown = [c for c in TIME_VARYING_UNKNOWN_REALS if c in pdf.columns]
    available_known = [c for c in TIME_VARYING_KNOWN_REALS if c in pdf.columns]

    # Build TimeSeriesDataSet
    dataset = TimeSeriesDataSet(
        pdf,
        time_idx="time_idx",
        target=target,
        group_ids=["ticker"],
        max_encoder_length=encoder_length,
        min_encoder_length=min_encoder_length,
        max_prediction_length=prediction_length,
        min_prediction_length=min_prediction_length,
        time_varying_unknown_reals=[target] + available_unknown,
        time_varying_known_reals=available_known,
        static_categoricals=STATIC_CATEGORICALS,
        categorical_encoders={"ticker": NaNLabelEncoder(add_nan=True)},
        target_normalizer=None,  # Keep target in cents (0-100)
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    logger.info(
        "Built TimeSeriesDataSet: %d samples, encoder=%d, prediction=%d, "
        "%d unknown reals, %d known reals",
        len(dataset),
        encoder_length,
        prediction_length,
        len(available_unknown),
        len(available_known),
    )
    return dataset


def create_dataloaders(
    df: pl.DataFrame,
    train_frac: float = 0.8,
    encoder_length: int = 60,
    prediction_length: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, TimeSeriesDataSet]:
    """Create train and validation dataloaders with proper time-series split.

    Splits data chronologically per ticker (no future leakage).

    Args:
        df: Feature-engineered DataFrame.
        train_frac: Fraction of data for training (remainder for validation).
        encoder_length: Number of historical time steps.
        prediction_length: Number of future time steps to predict.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        (train_dataloader, val_dataloader, training_dataset)
    """
    # Time-series split: use first train_frac of time_idx per ticker
    max_time_idx = df["time_idx"].max()
    train_cutoff = int(max_time_idx * train_frac)

    train_df = df.filter(pl.col("time_idx") <= train_cutoff)
    val_df = df.filter(pl.col("time_idx") > train_cutoff - encoder_length)

    logger.info(
        "Train/val split: cutoff=%d, train=%d rows, val=%d rows",
        train_cutoff, train_df.height, val_df.height,
    )

    # Build training dataset
    training = build_dataset(
        train_df,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
    )

    # Build validation dataset from training parameters
    val_pdf = val_df.to_pandas()
    val_pdf["time_idx"] = val_pdf["time_idx"].astype(int)
    val_pdf["ticker"] = val_pdf["ticker"].astype(str)

    validation = TimeSeriesDataSet.from_dataset(
        training, val_pdf, predict=True, stop_randomization=True
    )

    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=num_workers
    )

    logger.info(
        "Created dataloaders: train=%d batches, val=%d batches",
        len(train_dataloader), len(val_dataloader),
    )
    return train_dataloader, val_dataloader, training