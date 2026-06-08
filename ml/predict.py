"""
TFT batch inference — Stock Intelligence Platform.

Loads the best TFT checkpoint from GCS, runs multi-horizon
prediction (5 ticks ahead with 10/50/90th percentile bounds),
and writes results to BigQuery for the Flask dashboard.

Outputs per symbol:
  - predicted_price  (median / 50th pct)
  - lower_bound      (10th pct — bearish scenario)
  - upper_bound      (90th pct — bullish scenario)
  - direction        UP | DOWN
  - change_pct
  - attention_weights (which past ticks mattered most)

Usage:
    python predict.py                 # all symbols
    python predict.py --symbol AAPL   # single symbol
    python predict.py --horizon 3     # predict 3 ticks ahead
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from google.cloud import bigquery, storage
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from symbol_config import PROJECT_ID, BQ_FEAT_TABLE, BQ_PRED_TABLE, SYMBOLS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger     = logging.getLogger("tft_predict")
GCS_BUCKET = os.getenv("MODEL_BUCKET", f"{PROJECT_ID}-models")
bq_client  = bigquery.Client(project=PROJECT_ID)
gcs_client = storage.Client()


# ── Model loader ───────────────────────────────────────────────────────────────

def load_checkpoint(symbol: str) -> TemporalFusionTransformer:
    """Download the best TFT checkpoint from GCS and load it."""
    tmp = tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False)
    gcs_client.bucket(GCS_BUCKET).blob(
        f"stock/{symbol}/tft_best.ckpt"
    ).download_to_filename(tmp.name)
    model = TemporalFusionTransformer.load_from_checkpoint(tmp.name)
    model.eval()
    logger.info("Loaded TFT checkpoint for %s", symbol)
    return model


# ── Data preparation ───────────────────────────────────────────────────────────

def fetch_recent(symbol: str,
                 n: int = 120) -> pd.DataFrame:
    """Fetch last N ticks from BigQuery to build the encoder context."""
    sql = f"""
        SELECT tick_ts, price,
               COALESCE(volume, 0)         AS volume,
               COALESCE(vwap, price)       AS vwap,
               COALESCE(price_return, 0.0) AS price_return,
               COALESCE(price_usd_m, 0.0)  AS price_usd_m
        FROM `{BQ_FEAT_TABLE}`
        WHERE symbol = @symbol AND price > 0
        ORDER BY tick_ts DESC LIMIT {n}
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("symbol", "STRING", symbol)]
    )
    df = (bq_client.query(sql, job_config=cfg)
                   .to_dataframe()
                   .sort_values("tick_ts")
                   .reset_index(drop=True))
    df["time_idx"]   = range(len(df))
    df["symbol"]     = symbol
    df["hour"]       = df["tick_ts"].dt.hour.astype(str)
    df["dow"]        = df["tick_ts"].dt.dayofweek.astype(str)
    df["log_price"]  = np.log1p(df["price"])
    return df


def build_predict_dataset(df: pd.DataFrame,
                           training_ds: TimeSeriesDataSet,
                           max_prediction: int = 5) -> TimeSeriesDataSet:
    """Wrap the recent-data DataFrame as a prediction-mode dataset."""
    return TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(model: TemporalFusionTransformer,
                  df: pd.DataFrame,
                  symbol: str) -> dict:
    """Run TFT quantile prediction and extract interpretable outputs.

    Returns a dict with median forecast, bounds, direction, and
    top-5 attention weights (which encoder time steps the model
    attended to most).
    """
    # We need the training dataset schema to create a compatible predict ds.
    # Here we re-build a minimal training dataset from the same df.
    from pytorch_forecasting.data import GroupNormalizer
    training_ds = TimeSeriesDataSet(
        df[df["time_idx"] <= int(len(df) * 0.85)],
        time_idx="time_idx",
        target="log_price",
        group_ids=["symbol"],
        min_encoder_length=30,
        max_encoder_length=60,
        min_prediction_length=1,
        max_prediction_length=5,
        static_categoricals=["symbol"],
        time_varying_known_categoricals=["hour","dow"],
        time_varying_unknown_reals=["log_price","vwap","volume",
                                     "price_return","price_usd_m"],
        target_normalizer=GroupNormalizer(groups=["symbol"],
                                          transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    predict_ds = TimeSeriesDataSet.from_dataset(
        training_ds, df, predict=True, stop_randomization=True
    )
    predict_dl = predict_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    with torch.no_grad():
        raw_preds, x = model.predict(predict_dl, mode="raw", return_x=True)

    # Quantile outputs: shape [samples, time_steps, quantiles]
    # quantiles = [0.1, 0.5, 0.9]
    log_preds = raw_preds["prediction"][0].cpu().numpy()  # [horizon, 3]
    preds_inv = np.expm1(log_preds)                        # inverse log1p

    current_price = float(df["price"].iloc[-1])
    pred_median   = float(preds_inv[-1, 1])                # 5th tick, 50th pct
    pred_lower    = float(preds_inv[-1, 0])                # 10th pct
    pred_upper    = float(preds_inv[-1, 2])                # 90th pct
    change_pct    = round((pred_median - current_price) / current_price * 100, 4)

    # Attention weights — top-5 most attended encoder positions
    attn = model.interpret_output(raw_preds, reduction="sum")
    encoder_attention = attn.get("encoder_variables", {})
    top_attn = sorted(
        encoder_attention.items(), key=lambda kv: kv[1], reverse=True
    )[:5] if encoder_attention else []

    return {
        "symbol":           symbol,
        "current_price":    round(current_price, 4),
        "predicted_price":  round(pred_median, 4),
        "lower_bound":      round(pred_lower, 4),
        "upper_bound":      round(pred_upper, 4),
        "change_pct":       change_pct,
        "direction":        "UP" if pred_median > current_price else "DOWN",
        "model_type":       "temporal_fusion_transformer",
        "model_version":    "tft_v1",
        "top_attention":    json.dumps([{"feature": k, "weight": round(float(v), 4)}
                                        for k, v in top_attn]),
        "prediction_time":  datetime.now(timezone.utc).isoformat(),
        "created_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── BigQuery writer ────────────────────────────────────────────────────────────

def write_predictions(rows: list[dict]) -> None:
    errors = bq_client.insert_rows_json(BQ_PRED_TABLE, rows)
    if errors:
        logger.error("BigQuery write errors: %s", errors)
    else:
        logger.info("Wrote %d TFT predictions to %s", len(rows), BQ_PRED_TABLE)


# ── Entry point ────────────────────────────────────────────────────────────────

def run(symbols: list[str]) -> None:
    predictions = []
    for sym in symbols:
        try:
            model = load_checkpoint(sym)
            df    = fetch_recent(sym, n=120)
            pred  = run_inference(model, df, sym)
            predictions.append(pred)
            logger.info(
                "[%s] current=$%.2f  predicted=$%.2f [%.2f-%.2f]  %s  (%.2f%%)",
                sym, pred["current_price"], pred["predicted_price"],
                pred["lower_bound"], pred["upper_bound"],
                pred["direction"], pred["change_pct"],
            )
        except Exception as exc:
            logger.error("TFT prediction failed for %s: %s", sym, exc)

    if predictions:
        write_predictions(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT batch inference")
    parser.add_argument("--symbol", default=None)
    args   = parser.parse_args()
    target = [args.symbol.upper()] if args.symbol else SYMBOLS
    run(target)
