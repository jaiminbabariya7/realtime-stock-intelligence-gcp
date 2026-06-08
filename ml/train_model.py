"""
Temporal Fusion Transformer (TFT) training — Stock Intelligence Platform.

Fetches feature-engineered tick data from BigQuery, structures it as
a multi-variate time-series dataset, trains a TFT model via PyTorch
Forecasting, evaluates on a hold-out window, and saves the checkpoint
to GCS + logs metrics to MLflow.

TFT combines:
  - LSTM encoder for local pattern capture
  - Multi-head self-attention for long-range dependencies
  - Variable Selection Networks for automatic feature weighting
  - Quantile outputs for prediction intervals (10th, 50th, 90th pct)

References:
  Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting" — International Journal of
  Forecasting.

Usage:
    python train_model.py --symbol AAPL --horizon 5 --max_epochs 50
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from google.cloud import bigquery, storage
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.preprocessing import RobustScaler

from symbol_config import PROJECT_ID, BQ_FEAT_TABLE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("tft_trainer")

# ── Config from env ────────────────────────────────────────────────────────────
GCS_BUCKET     = os.getenv("MODEL_BUCKET",     f"{PROJECT_ID}-models")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MAX_PREDICTION = int(os.getenv("MAX_PREDICTION_LENGTH", "5"))    # ticks ahead
MAX_ENCODER    = int(os.getenv("MAX_ENCODER_LENGTH",    "60"))   # context window
BATCH_SIZE     = int(os.getenv("BATCH_SIZE",            "64"))
LEARNING_RATE  = float(os.getenv("LEARNING_RATE",       "3e-4"))

# ── Data loading ───────────────────────────────────────────────────────────────

def load_features(symbol: str, window: int = 5000) -> pd.DataFrame:
    """Fetch the last `window` feature-engineered ticks from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    sql = f"""
        SELECT
            tick_ts,
            price,
            volume,
            COALESCE(vwap, price)          AS vwap,
            COALESCE(price_return, 0.0)    AS price_return,
            COALESCE(price_usd_m, 0.0)     AS price_usd_m
        FROM `{BQ_FEAT_TABLE}`
        WHERE symbol = @symbol
          AND price > 0
        ORDER BY tick_ts DESC
        LIMIT {window}
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("symbol", "STRING", symbol)]
    )
    df = client.query(sql, job_config=cfg).to_dataframe()
    df = df.sort_values("tick_ts").reset_index(drop=True)
    logger.info("Loaded %d rows for %s", len(df), symbol)
    return df


def prepare_dataset(df: pd.DataFrame,
                    symbol: str) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """Build TFT-compatible TimeSeriesDataSet from a feature DataFrame.

    Returns (training_dataset, validation_dataset).
    """
    df = df.copy()
    df["time_idx"] = range(len(df))
    df["symbol"]   = symbol
    df["hour"]     = df["tick_ts"].dt.hour.astype(str)
    df["dow"]      = df["tick_ts"].dt.dayofweek.astype(str)

    # Log-scale price to stabilise gradients
    df["log_price"] = np.log1p(df["price"])

    val_cutoff = int(len(df) * 0.85)

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= val_cutoff],
        time_idx="time_idx",
        target="log_price",
        group_ids=["symbol"],
        min_encoder_length=MAX_ENCODER // 2,
        max_encoder_length=MAX_ENCODER,
        min_prediction_length=1,
        max_prediction_length=MAX_PREDICTION,
        static_categoricals=["symbol"],
        time_varying_known_categoricals=["hour", "dow"],
        time_varying_unknown_reals=["log_price", "vwap", "volume",
                                     "price_return", "price_usd_m"],
        target_normalizer=GroupNormalizer(groups=["symbol"],
                                          transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    return training, validation


# ── Model builder ──────────────────────────────────────────────────────────────

def build_tft(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Instantiate a TFT from the dataset's metadata."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )


# ── Artefact upload ────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    storage.Client().bucket(GCS_BUCKET).blob(gcs_path).upload_from_filename(local_path)
    logger.info("Uploaded %s → gs://%s/%s", local_path, GCS_BUCKET, gcs_path)


# ── Training entry point ───────────────────────────────────────────────────────

def train(symbol: str, window: int = 5000, max_epochs: int = 50) -> None:
    """Full training pipeline: data → TFT → evaluation → GCS upload."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("stock-tft")

    with mlflow.start_run(run_name=f"tft_{symbol}"):
        mlflow.log_params({
            "symbol": symbol, "window": window,
            "max_encoder_length": MAX_ENCODER,
            "max_prediction_length": MAX_PREDICTION,
            "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        })

        # 1. Data
        df = load_features(symbol, window)
        training_ds, val_ds = prepare_dataset(df, symbol)
        train_dl = training_ds.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=2)
        val_dl   = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=2)

        # 2. Model
        model = build_tft(training_ds)
        logger.info("TFT parameters: %d", sum(p.numel() for p in model.parameters()))

        # 3. Callbacks
        ckpt_dir = f"models/{symbol}"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, mode="min"),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=ckpt_dir, filename="tft_best",
                            monitor="val_loss", save_top_k=1, mode="min"),
        ]

        # 4. Train
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            logger=MLFlowLogger(experiment_name="stock-tft",
                                tracking_uri=MLFLOW_URI),
        )
        trainer.fit(model, train_dl, val_dl)

        # 5. Evaluate
        best_path = f"{ckpt_dir}/tft_best.ckpt"
        best      = TemporalFusionTransformer.load_from_checkpoint(best_path)
        preds     = best.predict(val_dl, return_y=True, trainer_kwargs={"logger": False})
        mae_val   = MAE()(preds.output, preds.y).item()
        rmse_val  = RMSE()(preds.output, preds.y).item()
        logger.info("[%s] val MAE=%.6f  RMSE=%.6f", symbol, mae_val, rmse_val)
        mlflow.log_metrics({"val_mae": mae_val, "val_rmse": rmse_val})

        # 6. Upload checkpoint + dataset metadata to GCS
        upload_to_gcs(best_path, f"stock/{symbol}/tft_best.ckpt")
        mlflow.pytorch.log_model(best, "tft_model")
        logger.info("Training complete for %s.", symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TFT for stock forecasting")
    parser.add_argument("--symbol",     default="AAPL")
    parser.add_argument("--window",     type=int, default=5000)
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()
    train(args.symbol, args.window, args.max_epochs)
