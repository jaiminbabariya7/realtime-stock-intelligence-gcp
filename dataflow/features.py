"""
Pure feature-engineering logic for the stock pipeline.

Kept free of Apache Beam imports so it can be unit-tested directly; the Beam
DoFns in stock_pipeline.py call these functions and hold the per-symbol state.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"symbol", "price", "volume", "timestamp"}


def parse_tick(raw: bytes) -> dict | None:
    """Decode a Pub/Sub message into a tick dict, or None if it is invalid."""
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Parse error: %s | raw=%s", exc, raw[:200])
        return None
    if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record):
        logger.warning("Dropping record: missing required fields")
        return None
    try:
        record["price"] = float(record["price"])
        record["volume"] = int(record["volume"])
    except (TypeError, ValueError):
        return None
    if record["price"] <= 0 or record["volume"] < 0:
        return None
    return record


def empty_state() -> dict:
    return {"prices": [], "volumes": []}


def compute_features(record: dict, state: dict | None,
                     sma_short: int, sma_long: int) -> tuple[dict, dict]:
    """Compute rolling features for one tick of one symbol.

    Args:
        record: parsed tick (symbol, price, volume, timestamp, ...).
        state: this symbol's rolling history from previous ticks (or None).
        sma_short / sma_long: moving-average window lengths in ticks;
            sma_long also bounds how much history is kept.

    Returns:
        (feature row for BigQuery, updated state).
    """
    state = state or empty_state()
    price, vol = record["price"], record["volume"]

    prices = (state["prices"] + [price])[-sma_long:]
    volumes = (state["volumes"] + [vol])[-sma_long:]
    n = len(prices)

    # Price return vs previous tick
    price_return = (price - prices[-2]) / prices[-2] if n >= 2 else None

    # VWAP over the rolling window
    total_vol = sum(volumes)
    vwap = (round(sum(p * v for p, v in zip(prices, volumes)) / total_vol, 4)
            if n >= 2 and total_vol > 0 else None)

    # Simple moving averages (only once enough history exists)
    sma_s = round(sum(prices[-sma_short:]) / sma_short, 4) if n >= sma_short else None
    sma_l = round(sum(prices) / sma_long, 4) if n >= sma_long else None

    row = {
        "symbol":       record["symbol"],
        "price":        price,
        "volume":       vol,
        "price_return": round(price_return, 6) if price_return is not None else None,
        "vwap":         vwap,
        "sma_short":    sma_s,
        "sma_long":     sma_l,
        "price_usd_m":  round(price * vol / 1_000_000, 6),   # dollar volume (millions)
        "exchange":     record.get("exchange", ""),
        "trade_id":     record.get("trade_id", ""),
        "tick_ts":      record.get("timestamp", ""),
        "window_end":   None,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    return row, {"prices": prices, "volumes": volumes}
