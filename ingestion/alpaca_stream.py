"""
Alpaca Markets live data ingestion → Google Cloud Pub/Sub.

Opens a WebSocket stream for all configured symbols and publishes
each trade tick as a JSON message to Pub/Sub. Handles reconnection
with exponential back-off and graceful shutdown on SIGTERM.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

from alpaca.data.live import StockDataStream
from google.cloud import pubsub_v1

from symbol_config import (
    SYMBOLS, PROJECT_ID, PUBSUB_TOPIC,
    RECONNECT_DELAY, MAX_RECONNECTS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("alpaca_stream")

# ── Alpaca credentials ────────────────────────────────────────────────────────
API_KEY    = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

# ── Pub/Sub client ────────────────────────────────────────────────────────────
publisher   = pubsub_v1.PublisherClient()
TOPIC_PATH  = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)

# ── State ─────────────────────────────────────────────────────────────────────
_running      = True
_reconnects   = 0
_ticks_sent   = 0


def _build_payload(trade: Any) -> bytes:
    """Serialise an Alpaca trade object to UTF-8 JSON bytes."""
    payload = {
        "symbol":     trade.symbol,
        "price":      float(trade.price),
        "volume":     int(trade.size),
        "trade_id":   str(trade.id) if hasattr(trade, "id") else "",
        "exchange":   getattr(trade, "exchange", ""),
        "timestamp":  trade.timestamp.isoformat()
                      if hasattr(trade.timestamp, "isoformat")
                      else str(trade.timestamp),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(payload).encode("utf-8")


def _on_trade(trade: Any) -> None:
    """Callback invoked for every incoming trade tick."""
    global _ticks_sent
    try:
        data    = _build_payload(trade)
        future  = publisher.publish(
            TOPIC_PATH,
            data=data,
            symbol=trade.symbol,
        )
        future.result(timeout=5)          # block briefly; raises on failure
        _ticks_sent += 1
        if _ticks_sent % 500 == 0:
            logger.info("Published %d ticks so far", _ticks_sent)
    except Exception as exc:
        logger.error("Failed to publish tick for %s: %s", trade.symbol, exc)


def _on_error(exc: Exception) -> None:
    logger.error("Stream error: %s", exc)


def _shutdown_handler(signum: int, frame: Any) -> None:
    global _running
    logger.info("Shutdown signal received — stopping stream.")
    _running = False
    sys.exit(0)


def run() -> None:
    """Main entry point — stream trades with reconnection logic."""
    global _reconnects, _running

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT,  _shutdown_handler)

    logger.info("Starting Alpaca stream | symbols=%s | topic=%s", SYMBOLS, TOPIC_PATH)

    while _running and _reconnects <= MAX_RECONNECTS:
        try:
            stream = StockDataStream(API_KEY, SECRET_KEY)
            stream.subscribe_trades(_on_trade, *SYMBOLS)
            logger.info("WebSocket connected. Streaming %d symbols...", len(SYMBOLS))
            stream.run()                  # blocks until disconnect
        except Exception as exc:
            _reconnects += 1
            wait = RECONNECT_DELAY * (2 ** min(_reconnects, 5))
            logger.warning(
                "Stream disconnected (%s). Reconnect %d/%d in %ds.",
                exc, _reconnects, MAX_RECONNECTS, wait,
            )
            time.sleep(wait)

    logger.info("Stream terminated. Total ticks published: %d", _ticks_sent)


if __name__ == "__main__":
    run()
