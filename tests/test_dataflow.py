"""Unit tests for the Dataflow pipeline's parsing and feature logic.

The logic lives in dataflow/features.py (no Apache Beam import), so these
tests exercise the exact functions the Beam DoFns call.
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataflow"))

from features import compute_features, empty_state, event_time_seconds, parse_tick  # noqa: E402


def _tick(price, volume=100, symbol="AAPL"):
    return {"symbol": symbol, "price": price, "volume": volume,
            "timestamp": "2024-07-15T09:31:00Z"}


def _run(prices, volumes=None, sma_short=3, sma_long=5):
    """Feed a sequence of ticks through compute_features, carrying state."""
    volumes = volumes or [100] * len(prices)
    state, rows = None, []
    for p, v in zip(prices, volumes):
        row, state = compute_features(_tick(p, v), state, sma_short, sma_long)
        rows.append(row)
    return rows, state


class TestParseTick(unittest.TestCase):
    def test_valid_message_parsed_and_cast(self):
        raw = json.dumps({"symbol": "TSLA", "price": "248.50", "volume": "5000",
                          "timestamp": "2024-07-15T09:31:00Z"}).encode()
        rec = parse_tick(raw)
        self.assertEqual(rec["symbol"], "TSLA")
        self.assertEqual(rec["price"], 248.5)
        self.assertEqual(rec["volume"], 5000)

    def test_malformed_json_dropped(self):
        self.assertIsNone(parse_tick(b"not-json"))

    def test_missing_fields_dropped(self):
        self.assertIsNone(parse_tick(json.dumps({"symbol": "TSLA", "price": 1}).encode()))

    def test_zero_or_negative_price_dropped(self):
        raw = json.dumps({"symbol": "TSLA", "price": 0, "volume": 10,
                          "timestamp": "t"}).encode()
        self.assertIsNone(parse_tick(raw))


class TestEventTime(unittest.TestCase):
    def test_iso_with_z_and_offset(self):
        self.assertEqual(event_time_seconds("1970-01-01T00:01:00Z"), 60.0)
        self.assertEqual(event_time_seconds("1970-01-01T01:01:00+01:00"), 60.0)

    def test_unparseable_returns_none(self):
        self.assertIsNone(event_time_seconds("not-a-time"))


class TestComputeFeatures(unittest.TestCase):
    def test_first_tick_has_no_history_features(self):
        row, state = compute_features(_tick(100.0), None, 3, 5)
        self.assertIsNone(row["price_return"])
        self.assertIsNone(row["vwap"])
        self.assertIsNone(row["sma_short"])
        self.assertEqual(state["prices"], [100.0])

    def test_price_return(self):
        rows, _ = _run([100.0, 105.0])
        self.assertAlmostEqual(rows[-1]["price_return"], 0.05)

    def test_vwap_weights_by_volume(self):
        rows, _ = _run([100.0, 110.0], volumes=[100, 300])
        self.assertAlmostEqual(rows[-1]["vwap"], (100 * 100 + 110 * 300) / 400)

    def test_sma_short_and_long(self):
        rows, _ = _run([100.0, 102.0, 104.0, 103.0, 105.0])
        self.assertIsNone(rows[1]["sma_short"])              # only 2 ticks so far
        self.assertAlmostEqual(rows[2]["sma_short"], 102.0)  # mean(100, 102, 104)
        self.assertIsNone(rows[3]["sma_long"])               # only 4 ticks so far
        self.assertAlmostEqual(rows[4]["sma_long"], 102.8)   # mean of all 5
        self.assertAlmostEqual(rows[4]["sma_short"], 104.0)  # mean(104, 103, 105)

    def test_history_bounded_to_sma_long(self):
        _, state = _run([float(p) for p in range(100, 120)], sma_long=5)
        self.assertEqual(state["prices"], [115.0, 116.0, 117.0, 118.0, 119.0])
        self.assertEqual(len(state["volumes"]), 5)

    def test_input_state_not_mutated(self):
        state = empty_state()
        compute_features(_tick(100.0), state, 3, 5)
        self.assertEqual(state, empty_state())

    def test_dollar_volume_in_millions(self):
        row, _ = compute_features(_tick(250.0, 4000), None, 3, 5)
        self.assertAlmostEqual(row["price_usd_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
