"""Runs the real Beam transform chain (dataflow/stock_pipeline.enrich) on the
DirectRunner. Skipped when apache-beam is not installed."""
import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "ingestion"))
sys.path.insert(0, os.path.join(ROOT, "dataflow"))
os.environ.setdefault("GCP_PROJECT_ID", "test-project")

try:
    import apache_beam as beam
    from apache_beam.testing.util import assert_that
except ImportError:                                  # pragma: no cover
    beam = None


def _msg(symbol, price, ts):
    return json.dumps({"symbol": symbol, "price": price, "volume": 100,
                       "timestamp": f"2024-07-15T09:30:{ts:02d}Z"}).encode()


@unittest.skipIf(beam is None, "apache-beam not installed")
class TestEnrichPipeline(unittest.TestCase):
    def test_state_is_kept_per_symbol(self):
        from stock_pipeline import enrich

        msgs = [_msg("AAPL", 100.0, 1), _msg("MSFT", 400.0, 2), _msg("AAPL", 110.0, 3),
                b"not-json", _msg("AAPL", 120.0, 4)]

        def check(rows):
            aapl = sorted((r for r in rows if r["symbol"] == "AAPL"), key=lambda r: r["price"])
            msft = [r for r in rows if r["symbol"] == "MSFT"]
            assert len(rows) == 4, rows                          # bad message dropped
            assert aapl[0]["price_return"] is None                # first AAPL tick
            assert abs(aapl[1]["price_return"] - 0.10) < 1e-9     # 100 -> 110, not vs MSFT
            assert aapl[2]["sma_short"] == 110.0                  # mean(100, 110, 120)
            assert msft[0]["price_return"] is None                # MSFT has its own history
            assert all(r["window_end"] for r in rows)

        with beam.Pipeline() as p:
            out = enrich(p | beam.Create(msgs), sma_short=3, sma_long=5)
            assert_that(out, check)


if __name__ == "__main__":
    unittest.main()
