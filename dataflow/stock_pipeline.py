"""
Apache Beam streaming pipeline — Stock Intelligence Platform.

Reads raw trade ticks from Pub/Sub, computes technical indicators
(SMA, VWAP, price return), validates records, and writes to BigQuery.

Deploy: python stock_pipeline.py --runner=DataflowRunner ...
Local:  python stock_pipeline.py --runner=DirectRunner ...
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Iterator

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions, PipelineOptions, StandardOptions,
)
from apache_beam.transforms.window import FixedWindows
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition

from symbol_config import (
    PROJECT_ID, PUBSUB_SUB, BQ_FEAT_TABLE,
    SMA_SHORT, SMA_LONG,
)

logger = logging.getLogger(__name__)

# ── BigQuery schema ────────────────────────────────────────────────────────────
BQ_SCHEMA = {
    "fields": [
        {"name": "symbol",       "type": "STRING",    "mode": "REQUIRED"},
        {"name": "price",        "type": "FLOAT64",   "mode": "REQUIRED"},
        {"name": "volume",       "type": "INTEGER",   "mode": "REQUIRED"},
        {"name": "price_return", "type": "FLOAT64",   "mode": "NULLABLE"},
        {"name": "vwap",         "type": "FLOAT64",   "mode": "NULLABLE"},
        {"name": "price_usd_m",  "type": "FLOAT64",   "mode": "NULLABLE"},
        {"name": "exchange",     "type": "STRING",    "mode": "NULLABLE"},
        {"name": "trade_id",     "type": "STRING",    "mode": "NULLABLE"},
        {"name": "tick_ts",      "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "window_end",   "type": "TIMESTAMP", "mode": "NULLABLE"},
        {"name": "processed_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}

# Per-symbol running state (worker-local, keyed by symbol)
_price_history: dict[str, list[float]]  = {}
_vol_history:   dict[str, list[float]]  = {}


# ── Transforms ─────────────────────────────────────────────────────────────────

class ParseAndValidate(beam.DoFn):
    """Decode Pub/Sub bytes → dict; drop malformed messages."""

    REQUIRED_FIELDS = {"symbol", "price", "volume", "timestamp"}

    def process(self, element: bytes) -> Iterator[dict]:
        try:
            record = json.loads(element.decode("utf-8"))
            if not self.REQUIRED_FIELDS.issubset(record):
                logger.warning("Dropping record — missing fields: %s",
                               self.REQUIRED_FIELDS - set(record))
                return
            record["price"]  = float(record["price"])
            record["volume"] = int(record["volume"])
            if record["price"] <= 0 or record["volume"] < 0:
                return
            yield record
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.error("Parse error: %s | raw=%s", exc, element[:200])


class FeatureEngineering(beam.DoFn):
    """Compute per-symbol rolling technical features (worker-local state)."""

    def process(self, record: dict) -> Iterator[dict]:
        sym   = record["symbol"]
        price = record["price"]
        vol   = record["volume"]

        # Rolling price and volume history (bounded to SMA_LONG)
        ph = _price_history.setdefault(sym, [])
        vh = _vol_history.setdefault(sym, [])
        ph.append(price)
        vh.append(vol)
        if len(ph) > SMA_LONG:
            ph.pop(0)
            vh.pop(0)

        n = len(ph)

        # Price return vs previous tick
        price_return = ((price - ph[-2]) / ph[-2]) if n >= 2 else None

        # VWAP (volume-weighted average price over rolling window)
        if n >= 2:
            notional = sum(p * v for p, v in zip(ph, vh))
            total_vol = sum(vh)
            vwap = round(notional / total_vol, 4) if total_vol > 0 else None
        else:
            vwap = None

        # Dollar volume (millions)
        price_usd_m = round(price * vol / 1_000_000, 6)

        yield {
            "symbol":       sym,
            "price":        price,
            "volume":       vol,
            "price_return": round(price_return, 6) if price_return is not None else None,
            "vwap":         vwap,
            "price_usd_m":  price_usd_m,
            "exchange":     record.get("exchange", ""),
            "trade_id":     record.get("trade_id", ""),
            "tick_ts":      record.get("timestamp", ""),
            "window_end":   None,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }


class AddWindowEnd(beam.DoFn):
    """Stamp each record with the window boundary time."""

    def process(self, record: dict,
                window=beam.DoFn.WindowParam) -> Iterator[dict]:
        end_dt = datetime.fromtimestamp(window.end, tz=timezone.utc)
        record["window_end"] = end_dt.isoformat()
        yield record


# ── Pipeline runner ────────────────────────────────────────────────────────────

def build_pipeline(p: beam.Pipeline) -> None:
    """Attach all transforms to the pipeline object."""
    (
        p
        | "ReadPubSub"      >> beam.io.ReadFromPubSub(subscription=PUBSUB_SUB)
        | "Window10s"       >> beam.WindowInto(FixedWindows(10))
        | "Parse"           >> beam.ParDo(ParseAndValidate())
        | "Features"        >> beam.ParDo(FeatureEngineering())
        | "AddWindow"       >> beam.ParDo(AddWindowEnd())
        | "WriteBigQuery"   >> WriteToBigQuery(
            BQ_FEAT_TABLE,
            schema=BQ_SCHEMA,
            write_disposition=BigQueryDisposition.WRITE_APPEND,
            create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
        )
    )


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stock Intelligence Dataflow pipeline")
    parser.add_argument("--gcs_temp",    default=f"gs://{PROJECT_ID}-temp/dataflow")
    parser.add_argument("--gcs_staging", default=f"gs://{PROJECT_ID}-staging/dataflow")
    parser.add_argument("--region",      default="us-central1")
    parser.add_argument("--machine_type",default="n1-standard-4")
    known, beam_args = parser.parse_known_args(argv)

    options = PipelineOptions(
        beam_args,
        streaming=True,
        save_main_session=True,
    )
    options.view_as(StandardOptions).streaming = True
    gcp_opts = options.view_as(GoogleCloudOptions)
    gcp_opts.project         = PROJECT_ID
    gcp_opts.region          = known.region
    gcp_opts.temp_location   = known.gcs_temp
    gcp_opts.staging_location= known.gcs_staging

    with beam.Pipeline(options=options) as p:
        build_pipeline(p)

    logger.info("Pipeline submitted.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
