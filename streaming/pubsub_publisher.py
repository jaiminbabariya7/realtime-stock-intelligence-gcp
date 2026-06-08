"""
Pub/Sub publisher utility — replays historical data or simulates live ticks.

Useful for local testing when Alpaca WebSocket is not available.
Reads price CSVs from GCS or generates synthetic data, publishes to Pub/Sub.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone

from google.cloud import pubsub_v1

from symbol_config import PROJECT_ID, PUBSUB_TOPIC, SYMBOLS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger    = logging.getLogger("pubsub_publisher")
publisher = pubsub_v1.PublisherClient()
TOPIC     = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)

# Synthetic seed prices for simulation
_BASE_PRICES: dict[str, float] = {
    "AAPL": 189.0, "TSLA": 248.0, "NVDA": 875.0,
    "MSFT": 415.0, "GOOGL": 175.0, "AMZN": 185.0,
    "META": 510.0, "NFLX": 630.0,
}


def _simulate_tick(symbol: str) -> dict:
    """Generate a realistic synthetic tick using random walk."""
    base  = _BASE_PRICES.get(symbol, 100.0)
    drift = random.gauss(0, base * 0.002)         # ±0.2% noise
    price = max(base + drift, 1.0)
    _BASE_PRICES[symbol] = price                   # walk forward
    return {
        "symbol":      symbol,
        "price":       round(price, 4),
        "volume":      random.randint(100, 5000),
        "exchange":    random.choice(["NYSE", "NASDAQ", "IEX"]),
        "trade_id":    f"sim_{int(time.time()*1000)}",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "source":      "simulator",
    }


def publish_ticks(rate_per_sec: int = 5,
                  total: int | None = None) -> None:
    """Publish simulated ticks to Pub/Sub.

    Args:
        rate_per_sec: Messages per second across all symbols.
        total: Stop after this many messages (None = run indefinitely).
    """
    sent = 0
    logger.info("Publishing simulated ticks to %s at %d/s ...", TOPIC, rate_per_sec)
    while total is None or sent < total:
        for sym in SYMBOLS:
            tick = _simulate_tick(sym)
            data = json.dumps(tick).encode("utf-8")
            publisher.publish(TOPIC, data=data, symbol=sym).result()
            sent += 1
        if sent % 100 == 0:
            logger.info("Published %d ticks", sent)
        time.sleep(1 / rate_per_sec)
    logger.info("Done. Total: %d", sent)


if __name__ == "__main__":
    publish_ticks(rate_per_sec=5)
