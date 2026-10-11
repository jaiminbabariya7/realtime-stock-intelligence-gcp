"""
Apache Beam streaming pipeline — Stock Intelligence Platform.

Reads raw trade ticks from Pub/Sub, validates records, computes per-symbol
features (price return, rolling VWAP, SMA short/long, dollar volume) using
Beam state keyed by symbol, windows them into 10-second windows, and writes
to BigQuery.

Deploy: python stock_pipeline.py --runner=DataflowRunner ...
Local:  python stock_pipeline.py --runner=DirectRunner ...
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Iterator

import apache_beam as beam
from apache_beam.coders import PickleCoder
from apache_beam.options.pipeline_options import (
    GoogleCloudOptions, PipelineOptions, SetupOptions, StandardOptions,
)
from apache_beam.transforms.window import FixedWindows
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition
from apache_beam.transforms.userstate import ReadModifyWriteStateSpec

from features import compute_features, parse_tick

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
        {"name": "sma_short",    "type": "FLOAT64",   "mode": "NULLABLE"},
        {"name": "sma_long",     "type": "FLOAT64",   "mode": "NULLABLE"},
        {"name": "price_usd_m",  "type": "FLOAT64",   "mode": "NULLABLE"},
        {"name": "exchange",     "type": "STRING",    "mode": "NULLABLE"},
        {"name": "trade_id",     "type": "STRING",    "mode": "NULLABLE"},
        {"name": "tick_ts",      "type": "TIMESTAMP", "mode": "REQUIRED"},
        {"name": "window_end",   "type": "TIMESTAMP", "mode": "NULLABLE"},
        {"name": "processed_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}

# ── Transforms ─────────────────────────────────────────────────────────────────

class ParseAndValidate(beam.DoFn):
    """Decode Pub/Sub bytes → dict; drop malformed messages."""

    def process(self, element: bytes) -> Iterator[dict]:
        record = parse_tick(element)
        if record is not None:
            yield record


class FeatureEngineering(beam.DoFn):
    """Compute per-symbol rolling features with Beam state.

    Input is keyed by symbol, so each symbol's price/volume history lives in
    runner-managed state: it is checkpointed with the pipeline, survives worker
    restarts, and is never split across workers.
    """

    HISTORY = ReadModifyWriteStateSpec("history", PickleCoder())

    def __init__(self, sma_short: int, sma_long: int) -> None:
        self.sma_short = sma_short
        self.sma_long = sma_long

    def process(self, element: tuple[str, dict],
                history=beam.DoFn.StateParam(HISTORY)) -> Iterator[dict]:
        _symbol, record = element
        row, new_state = compute_features(record, history.read(),
                                          self.sma_short, self.sma_long)
        history.write(new_state)
        yield row


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
        | "Parse"           >> beam.ParDo(ParseAndValidate())
        | "KeyBySymbol"     >> beam.Map(lambda r: (r["symbol"], r))
        # Stateful features run in the global window so history carries
        # across the 10-second windows applied afterwards.
        | "Features"        >> beam.ParDo(FeatureEngineering(SMA_SHORT, SMA_LONG))
        | "Window10s"       >> beam.WindowInto(FixedWindows(10))
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
    # Ship features.py to the Dataflow workers
    options.view_as(SetupOptions).setup_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "setup.py")
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
