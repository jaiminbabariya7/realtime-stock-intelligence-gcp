"""
LSTM + ARIMA hybrid model training for stock price forecasting.

Fetches feature-engineered data from BigQuery, trains an ARIMA model
on the residuals, trains an LSTM on the sequence data, and saves
both artefacts to GCS for serving by predict.py.

Usage:
    python train_model.py --symbol AAPL --window 3000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from symbol_config import PROJECT_ID, BQ_FEAT_TABLE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_model")

SEQ_LEN      = int(os.getenv("SEQ_LEN",      "20"))
ARIMA_ORDER  = (int(os.getenv("ARIMA_P", "5")),
                int(os.getenv("ARIMA_D", "1")),
                int(os.getenv("ARIMA_Q", "0")))
GCS_BUCKET   = os.getenv("MODEL_BUCKET", f"{PROJECT_ID}-models")
TEST_SPLIT   = float(os.getenv("TEST_SPLIT", "0.1"))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(symbol: str, window: int) -> pd.DataFrame:
    """Fetch the last `window` ticks for `symbol` from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    sql = f"""
        SELECT tick_ts, price, volume, vwap, price_return
        FROM `{BQ_FEAT_TABLE}`
        WHERE symbol = @symbol
        ORDER BY tick_ts DESC
        LIMIT {window}
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", symbol)
        ]
    )
    df = client.query(sql, job_config=cfg).to_dataframe()
    df = df.sort_values("tick_ts").reset_index(drop=True)
    logger.info("Loaded %d rows for %s from BigQuery.", len(df), symbol)
    return df


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(prices_scaled: np.ndarray,
                    seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Create (X, y) sliding-window sequences for LSTM."""
    X, y = [], []
    for i in range(seq_len, len(prices_scaled)):
        X.append(prices_scaled[i - seq_len:i])
        y.append(prices_scaled[i])
    return np.array(X), np.array(y)


# ── ARIMA ──────────────────────────────────────────────────────────────────────

def train_arima(prices: np.ndarray) -> tuple:
    """Fit ARIMA on the price series; return (model_fit, residuals)."""
    logger.info("Fitting ARIMA%s ...", ARIMA_ORDER)
    model_fit = ARIMA(prices, order=ARIMA_ORDER).fit()
    logger.info("ARIMA AIC=%.2f", model_fit.aic)
    return model_fit, model_fit.resid.values


# ── LSTM ───────────────────────────────────────────────────────────────────────

def build_lstm(seq_len: int) -> Sequential:
    """Build a stacked LSTM with dropout and batch normalisation."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse",
                  metrics=["mae"])
    return model


def train_lstm(residuals: np.ndarray,
               scaler: MinMaxScaler,
               seq_len: int,
               symbol: str) -> Sequential:
    """Scale residuals, build sequences, train LSTM, return fitted model."""
    res_scaled = scaler.fit_transform(residuals.reshape(-1, 1)).flatten()
    X, y       = build_sequences(res_scaled, seq_len)
    X          = X.reshape(X.shape[0], X.shape[1], 1)

    split  = int(len(X) * (1 - TEST_SPLIT))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = build_lstm(seq_len)
    cbs   = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ]
    logger.info("Training LSTM on %d samples (val=%d)...", len(X_tr), len(X_te))
    model.fit(X_tr, y_tr, epochs=100, batch_size=32,
              validation_data=(X_te, y_te), callbacks=cbs, verbose=0)

    # Evaluate
    y_pred = model.predict(X_te, verbose=0).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_te_inv   = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(y_te_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_te_inv, y_pred_inv))
    logger.info("[%s] LSTM MAE=%.4f  RMSE=%.4f", symbol, mae, rmse)
    return model


# ── Artefact upload ────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    bucket = storage.Client().bucket(GCS_BUCKET)
    bucket.blob(gcs_path).upload_from_filename(local_path)
    logger.info("Uploaded %s → gs://%s/%s", local_path, GCS_BUCKET, gcs_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def train(symbol: str, window: int) -> None:
    df     = load_data(symbol, window)
    prices = df["price"].values.astype(float)

    price_scaler = MinMaxScaler()
    prices_norm  = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    arima_fit, residuals = train_arima(prices_norm)

    res_scaler = MinMaxScaler()
    lstm_model = train_lstm(residuals, res_scaler, SEQ_LEN, symbol)

    # Save locally then upload
    Path("models").mkdir(exist_ok=True)
    arima_path  = f"models/{symbol}_arima.pkl"
    scaler_path = f"models/{symbol}_price_scaler.pkl"
    res_sc_path = f"models/{symbol}_res_scaler.pkl"
    lstm_path   = f"models/{symbol}_lstm.keras"
    meta_path   = f"models/{symbol}_meta.json"

    arima_fit.save(arima_path)
    joblib.dump(price_scaler, scaler_path)
    joblib.dump(res_scaler,   res_sc_path)
    lstm_model.save(lstm_path)

    meta = {"symbol": symbol, "seq_len": SEQ_LEN,
            "arima_order": list(ARIMA_ORDER), "window": window}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    for local, gcs in [
        (arima_path,  f"stock/{symbol}/arima.pkl"),
        (scaler_path, f"stock/{symbol}/price_scaler.pkl"),
        (res_sc_path, f"stock/{symbol}/res_scaler.pkl"),
        (lstm_path,   f"stock/{symbol}/lstm.keras"),
        (meta_path,   f"stock/{symbol}/meta.json"),
    ]:
        upload_to_gcs(local, gcs)

    logger.info("Training complete for %s.", symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--window", type=int, default=3000)
    args = parser.parse_args()
    train(args.symbol, args.window)
