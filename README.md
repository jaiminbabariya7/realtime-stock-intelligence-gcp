# Stock Intelligence Platform

## Overview

This project is an end-to-end **real-time data engineering and machine learning system** built on Google Cloud Platform.

It ingests **live stock market data from external API**, processes it through a **streaming ETL pipeline**, stores it in a **cloud data warehouse**, trains a **machine learning model**, and serves predictions via a **Flask App**.

The system demonstrates a complete modern data stack:

- Real-time data ingestion (multi-stock streaming)
- Cloud-based ETL pipeline
- Data warehouse modeling
- Feature engineering for time-series data
- Machine learning prediction system
- Web-based prediction dashboard

---

## Architecture

Live Stock API (Alpaca)
        ↓
Python Streaming Ingestion
        ↓
Google Cloud Pub/Sub
        ↓
Apache Beam / Dataflow (Streaming ETL)
        ↓
Google BigQuery (Data Warehouse)
        ↓
ML Training + Prediction Layer
        ↓
Flask Dashboard (UI)

---

## Tech Stack

### Cloud
- Google Cloud Platform (Pub/Sub, Dataflow, BigQuery, Cloud Storage, Cloud Run)

---

### Streaming & Data Ingestion
- Pub/Sub real-time messaging
- Alpaca API (live stock market data ingestion)

---

### Data Processing
- Apache Beam (Dataflow pipelines)
- Streaming ETL for cleaning, transformation, and feature engineering

---

### Machine Learning (Time-Series Forecasting)

This project uses **forecasting-based ML model**:

- ARIMA (statistical time-series forecasting)
- LSTM (deep learning sequence model using TensorFlow/Keras)
- Pandas, NumPy
- Scikit-learn (data preprocessing)
- Joblib (model serialization)

Objective: Predict **future stock prices** per symbol

---

### Backend & UI

- Flask (REST API + backend server)
- HTML / CSS (frontend dashboard)

---

## Data Engineering Layer

- Multi-stock real-time ingestion system
- Symbol rotation-based API ingestion (rate-limit safe design)
- Streaming ETL pipeline using Dataflow
- Structured storage in BigQuery (time-series optimized dataset)

---

## Machine Learning Layer

### Feature Engineering:
- Lag features (previous price values)
- Rolling averages
- Trend smoothing
- Price difference calculations

### Models:
- ARIMA → Statistical baseline forecasting model
- LSTM → Deep learning-based sequence forecasting model

### Output:
- Predicted future stock price per symbol

---

## Dashboard (Flask Application)

- Deployed through Cloud Run
- Search-based stock prediction system
- User inputs stock ticker (e.g., AAPL, TSLA, NVDA)
- Returns:
  - Predicted future stock price
  - Real-time results from ML models
  - Clean and minimal UI for demonstration purposes

---

## Future Improvements

- Add Kafka alternative streaming mode
- Integrate live chart visualization (Plotly / TradingView)
- Add model retraining scheduler (Airflow/Composer)
- Add multi-user authentication dashboard
