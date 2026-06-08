"""
Batch prediction runner — Stock Intelligence Platform.

Loads trained ARIMA + LSTM models from GCS, generates next-tick
price forecasts for each configured symbol, and writes predictions
to BigQuery for the Flask dashboard to query.

Usage:
    python predict.py                    # all symbols
    python predict.py --symbol AAPL      # single symbol
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
from tensorflow.keras.models import load_model

from symbol_config import PROJECT_ID, BQ_FEAT_TABLE, BQ_PRED_TABLE, SYMBOLS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("predict")

GCS_BUCKET = os.getenv("MODEL_BUCKET", f"{PROJECT_ID}-models")
SEQ_LEN    = int(os.getenv("SEQ_LEN", "20"))

bq_client  = bigquery.Client(project=PROJECT_ID)
gcs_client = storage.Client()


# ── Model loader ───────────────────────────────────────────────────────────────

def _download(gcs_path: str, suffix: str) -> str:
    """Download a GCS object to a temp file and return local path."""
    tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    blob = gcs_client.bucket(GCS_BUCKET).blob(gcs_path)
    blob.download_to_filename(tmp.name)
    return tmp.name


def load_artefacts(symbol: str) -> dict:
    """Download and load all model artefacts for a symbol."""
    logger.info("Loading models for %s from GCS...", symbol)
    arima_path  = _download(f"stock/{symbol}/arima.pkl",        ".pkl")
    scaler_path = _download(f"stock/{symbol}/price_scaler.pkl", ".pkl")
    res_sc_path = _download(f"stock/{symbol}/res_scaler.pkl",   ".pkl")
    lstm_path   = _download(f"stock/{symbol}/lstm.keras",       ".keras")
    meta_path   = _download(f"stock/{symbol}/meta.json",        ".json")

    with open(meta_path) as f:
        meta = json.load(f)

    return {
        "arima":         joblib.load(arima_path),
        "price_scaler":  joblib.load(scaler_path),
        "res_scaler":    joblib.load(res_sc_path),
        "lstm":          load_model(lstm_path),
        "seq_len":       meta.get("seq_len", SEQ_LEN),
    }


# ── Prediction logic ───────────────────────────────────────────────────────────

def fetch_recent_prices(symbol: str, n: int = 200) -> np.ndarray:
    """Fetch last N prices from BigQuery feature table."""
    sql = f"""
        SELECT price FROM `{BQ_FEAT_TABLE}`
        WHERE symbol = @symbol
        ORDER BY tick_ts DESC LIMIT {n}
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", symbol)
        ]
    )
    rows = list(bq_client.query(sql, job_config=cfg).result())
    if not rows:
        raise ValueError(f"No data found for symbol {symbol}")
    prices = np.array([r.price for r in reversed(rows)], dtype=float)
    return prices


def predict_next_price(symbol: str, arts: dict) -> dict:
    """Generate the next-tick hybrid ARIMA+LSTM price forecast.

    Returns a dict suitable for BigQuery insertion.
    """
    seq_len = arts["seq_len"]
    prices  = fetch_recent_prices(symbol, n=max(200, seq_len + 50))

    # 1. Scale prices
    p_scaled = arts["price_scaler"].transform(
        prices.reshape(-1, 1)
    ).flatten()

    # 2. ARIMA one-step forecast
    arima_fc    = arts["arima"].forecast(steps=1)[0]
    arima_resid = p_scaled[-1] - arima_fc     # residual at last known tick

    # 3. LSTM predicts next residual from recent residual history
    arima_resids = arts["arima"].resid.values[-seq_len:]
    res_scaled   = arts["res_scaler"].transform(
        arima_resids.reshape(-1, 1)
    ).flatten()
    X = res_scaled[-seq_len:].reshape(1, seq_len, 1)
    lstm_fc_scaled = arts["lstm"].predict(X, verbose=0)[0][0]
    lstm_fc_resid  = arts["res_scaler"].inverse_transform(
        [[lstm_fc_scaled]]
    )[0][0]

    # 4. Combine: hybrid forecast = ARIMA + LSTM residual correction
    hybrid_scaled = arima_fc + lstm_fc_resid
    predicted_price = float(
        arts["price_scaler"].inverse_transform([[hybrid_scaled]])[0][0]
    )
    current_price = float(prices[-1])
    change_pct    = round((predicted_price - current_price) / current_price * 100, 4)
    direction     = "UP" if predicted_price > current_price else "DOWN"

    now = datetime.now(timezone.utc).isoformat()
    return {
        "symbol":          symbol,
        "current_price":   round(current_price, 4),
        "predicted_price": round(predicted_price, 4),
        "change_pct":      change_pct,
        "direction":       direction,
        "model_version":   "arima_lstm_v1",
        "prediction_time": now,
        "created_at":      now,
    }


# ── Writer ─────────────────────────────────────────────────────────────────────

def write_predictions(rows: list[dict]) -> None:
    errors = bq_client.insert_rows_json(BQ_PRED_TABLE, rows)
    if errors:
        logger.error("BigQuery write errors: %s", errors)
    else:
        logger.info("Wrote %d predictions to %s", len(rows), BQ_PRED_TABLE)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(symbols: list[str]) -> None:
    predictions = []
    for sym in symbols:
        try:
            arts = load_artefacts(sym)
            pred = predict_next_price(sym, arts)
            predictions.append(pred)
            logger.info("[%s] current=$%.2f  predicted=$%.2f  (%s %.2f%%)",
                        sym, pred["current_price"], pred["predicted_price"],
                        pred["direction"], pred["change_pct"])
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", sym, exc)

    if predictions:
        write_predictions(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None,
                        help="Single symbol to predict (default: all)")
    args   = parser.parse_args()
    target = [args.symbol.upper()] if args.symbol else SYMBOLS
    run(target)
