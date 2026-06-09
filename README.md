# Real-Time Stock Intelligence Platform GCP

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Apache Beam](https://img.shields.io/badge/Apache%20Beam-2.54-orange?logo=apachebeam)
![GCP](https://img.shields.io/badge/Google%20Cloud-Pub%2FSub%20%7C%20Dataflow%20%7C%20BigQuery%20%7C%20Cloud%20Run-4285F4?logo=googlecloud)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![Docker](https://img.shields.io/badge/Docker-Cloud%20Run-2496ED?logo=docker)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> End-to-end real-time stock intelligence system on GCP. Live trade ticks stream from Alpaca Markets through Pub/Sub into an Apache Beam/Dataflow feature-engineering pipeline, land in BigQuery, feed an Temporal Fusion Transformer (TFT) forecasting model, and surface through a Flask dashboard deployed on Cloud Run.

---

## Table of Contents
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Components](#components)
- [BigQuery Schema](#bigquery-schema)
- [Setup](#setup)
- [Running Locally](#running-locally)
- [Deploying to GCP](#deploying-to-gcp)
- [ML Model](#ml-model)
- [API Endpoints](#api-endpoints)
- [Skills Demonstrated](#skills-demonstrated)

---

## Architecture

```
Alpaca Markets WebSocket (live trades: AAPL, TSLA, NVDA, MSFT, GOOGL ...)
        |
        v
[ingestion/alpaca_stream.py]  â reconnect logic, graceful shutdown
        |  JSON tick messages
        v
Google Cloud Pub/Sub  (topic: stock-price-events)
        |
        v
Apache Beam / Cloud Dataflow  [dataflow/stock_pipeline.py]
  |-- ParseAndValidate   â decode, validate, drop malformed ticks
  |-- FixedWindows(10s)  â 10-second tumbling windows
  |-- FeatureEngineering â price return, VWAP, dollar volume
  |-- AddWindowEnd       â stamp window boundary
        |
        v
BigQuery: stock_data.stock_features  (partitioned by tick_ts)
        |
        +----------> [ml/train_model.py]  â ARIMA + LSTM training â GCS
        |
        +----------> [ml/predict.py]      â batch forecasting â BigQuery predictions
        |
        v
BigQuery: stock_data.stock_predictions
        |
        v
Flask Dashboard  [flask_app/app.py]  â deployed on Cloud Run
  |-- /                       live dashboard (all symbols)
  |-- /api/latest/<symbol>    latest tick + prediction
  |-- /api/history/<symbol>   last 200 ticks for charting
  |-- /api/predictions        all symbol predictions
  |-- /health                 health check
```

---

## Project Structure

```
realtime-stock-intelligence-gcp/
|
|-- ingestion/
|   |-- alpaca_stream.py      # Alpaca WebSocket -> Pub/Sub (reconnect, SIGTERM handling)
|   |-- symbol_config.py      # Centralised config: symbols, topics, BQ tables, windows
|   |-- requirements.txt
|
|-- dataflow/
|   |-- stock_pipeline.py     # Apache Beam: parse -> window -> features -> BigQuery
|
|-- streaming/
|   |-- pubsub_publisher.py   # Simulated tick publisher for local testing
|
|-- ml/
|   |-- train_model.py        # ARIMA + stacked LSTM training; artefacts -> GCS
|   |-- predict.py            # Load models from GCS; batch predictions -> BigQuery
|   |-- requirements.txt
|
|-- flask_app/
|   |-- app.py                # 6-route Flask REST API + dashboard
|   |-- requirements.txt
|   |-- Dockerfile            # gunicorn on port 8080 for Cloud Run
|   |-- templates/
|   |   |-- index.html        # Live dashboard UI
|   |-- static/
|       |-- style.css
|
|-- tests/
|   |-- test_ingestion.py
|   |-- test_dataflow.py
|
|-- Snapshots/                # Screenshots: Dataflow graph, BigQuery, dashboard, Cloud Run
|-- pyproject.toml
|-- Makefile
|-- CONTRIBUTING.md
|-- .env.example
```

---

## Components

### 1. Ingestion â `ingestion/alpaca_stream.py`
Opens a WebSocket to Alpaca Markets for all configured symbols. Every trade tick is serialised to JSON and published to Pub/Sub. Features:
- Exponential back-off reconnection (up to `MAX_RECONNECTS`)
- Graceful SIGTERM/SIGINT shutdown â no lost ticks
- All config via environment variables (no hardcoded values)

### 2. Dataflow Pipeline â `dataflow/stock_pipeline.py`
Apache Beam streaming pipeline with:
- **ParseAndValidate** â decodes Pub/Sub bytes, drops malformed/zero-price ticks
- **FixedWindows(10s)** â groups ticks into 10-second tumbling windows
- **FeatureEngineering** â computes VWAP, price return (%), dollar volume ($ millions)
- **WriteToBigQuery** â appends enriched ticks to partitioned BigQuery table

### 3. ML Training â `ml/train_model.py`
Hybrid ARIMA + stacked LSTM model:
1. Fetches last N ticks per symbol from BigQuery feature table
2. Fits ARIMA on price series to model linear trend
3. Trains a 2-layer stacked LSTM on the ARIMA residuals (non-linear patterns)
4. Evaluates on 10% hold-out; logs MAE and RMSE
5. Saves all artefacts (model, scalers, metadata) to GCS

### 4. Prediction â `ml/predict.py`
Batch inference runner:
- Downloads model artefacts from GCS per symbol
- Combines ARIMA 1-step forecast + LSTM residual correction
- Writes `{symbol, current_price, predicted_price, change_pct, direction}` to BigQuery

### 5. Flask Dashboard â `flask_app/app.py`
REST API + web dashboard deployed on Cloud Run:

| Route | Description |
|---|---|
| `GET /` | Live monitoring dashboard |
| `GET /api/symbols` | List of monitored symbols |
| `GET /api/latest/<symbol>` | Latest tick + ML prediction |
| `GET /api/history/<symbol>` | Last 200 ticks (chart data) |
| `GET /api/predictions` | Latest predictions all symbols |
| `GET /health` | Health check |

---

## BigQuery Schema

```sql
-- stock_data.stock_features  (Dataflow writes here)
symbol        STRING    NOT NULL
price         FLOAT64   NOT NULL
volume        INTEGER   NOT NULL
price_return  FLOAT64               -- (price - prev_price) / prev_price
vwap          FLOAT64               -- volume-weighted average price
price_usd_m   FLOAT64               -- dollar volume in millions
exchange      STRING
trade_id      STRING
tick_ts       TIMESTAMP NOT NULL    -- PARTITION BY DATE(tick_ts)
window_end    TIMESTAMP
processed_at  TIMESTAMP NOT NULL

-- stock_data.stock_predictions  (predict.py writes here)
symbol          STRING
current_price   FLOAT64
predicted_price FLOAT64
change_pct      FLOAT64
direction       STRING    -- "UP" | "DOWN"
model_version   STRING
prediction_time TIMESTAMP
created_at      TIMESTAMP
```

---

## Setup

### Prerequisites
- GCP project with Pub/Sub, Dataflow, BigQuery, Cloud Run, GCS enabled
- Alpaca Markets account (free paper-trading account works)
- Python 3.11+

```bash
git clone https://github.com/jaiminbabariya7/realtime-stock-intelligence-gcp
cd realtime-stock-intelligence-gcp
cp .env.example .env
# Edit .env â fill in GCP_PROJECT_ID, Alpaca keys, bucket names
make install
```

---

## Running Locally

```bash
# 1. Simulate tick stream (no Alpaca account needed)
python streaming/pubsub_publisher.py

# 2. Run Dataflow pipeline locally
python dataflow/stock_pipeline.py --runner=DirectRunner

# 3. Train model for one symbol
python ml/train_model.py --symbol AAPL --window 3000

# 4. Run batch predictions
python ml/predict.py --symbol AAPL

# 5. Start dashboard
cd flask_app && flask run --port 8080
```

---

## Deploying to GCP

```bash
# Submit Dataflow pipeline
python dataflow/stock_pipeline.py \
  --runner=DataflowRunner \
  --region=us-central1 \
  --machine_type=n1-standard-4

# Deploy Flask dashboard to Cloud Run
cd flask_app
docker build -t gcr.io/$GCP_PROJECT_ID/stock-dashboard .
docker push gcr.io/$GCP_PROJECT_ID/stock-dashboard
gcloud run deploy stock-dashboard \
  --image gcr.io/$GCP_PROJECT_ID/stock-dashboard \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=$GCP_PROJECT_ID
```

---

## ML Model — Temporal Fusion Transformer (TFT)

The model is a **Temporal Fusion Transformer** (Lim et al., 2021 — *International Journal of Forecasting*), replacing traditional ARIMA with a modern architecture that captures both short-term patterns and long-range dependencies simultaneously.

```
Encoder Input (60 ticks):
  price, volume, vwap, price_return, price_usd_m   ← time-varying unknowns
  hour_of_day, day_of_week                          ← time-varying knowns
  symbol                                            ← static categorical
        |
        v
Variable Selection Network  ← weights each feature's contribution
        |
        v
LSTM Encoder (hidden_size=64)  ← captures local sequential patterns
        |
        v
Multi-Head Self-Attention (4 heads)  ← long-range dependencies
        |
        v
Quantile Output Head
  p10 (lower bound)   ← bearish scenario
  p50 (median)        ← point forecast used for direction signal
  p90 (upper bound)   ← bullish scenario
```

**Why TFT over ARIMA/plain LSTM:**
| | ARIMA | LSTM | TFT |
|---|---|---|---|
| Long-range dependencies | ✗ | partial | ✓ attention |
| Multi-variate inputs | limited | ✓ | ✓ |
| Prediction intervals | ✓ | ✗ | ✓ quantiles |
| Feature interpretability | ✗ | ✗ | ✓ variable weights |
| Multi-horizon output | ✗ | ✗ | ✓ |

**Training:** PyTorch Forecasting + PyTorch Lightning with EarlyStopping,
LR scheduling, and MLflow experiment tracking.
Each symbol's best checkpoint is saved to GCS; predictions expose median,
lower bound (p10), and upper bound (p90) with attention weight logging.

---

## Skills Demonstrated
`Python` Â· `Apache Beam` Â· `Cloud Dataflow` Â· `Google Pub/Sub` Â· `BigQuery` Â· `PyTorch` Â· `TFT` Â· `MLflow` Â· `Flask` Â· `Docker` Â· `Cloud Run` Â· `GCS` Â· `Real-Time Pipelines` Â· `MLOps`
