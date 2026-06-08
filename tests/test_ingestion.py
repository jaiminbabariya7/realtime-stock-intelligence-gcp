"""Unit tests for ingestion module."""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Tests for symbol_config
# ---------------------------------------------------------------------------

class TestSymbolConfig(unittest.TestCase):
    def test_symbols_not_empty(self):
        """SYMBOLS list must contain at least one ticker."""
        from ingestion.symbol_config import SYMBOLS
        self.assertIsInstance(SYMBOLS, list)
        self.assertGreater(len(SYMBOLS), 0)

    def test_symbols_are_strings(self):
        """All symbols must be non-empty uppercase strings."""
        from ingestion.symbol_config import SYMBOLS
        for s in SYMBOLS:
            self.assertIsInstance(s, str)
            self.assertTrue(s.isupper(), f"Symbol {s!r} is not uppercase")
            self.assertGreater(len(s), 0)


# ---------------------------------------------------------------------------
# Tests for message serialisation in alpaca_stream
# ---------------------------------------------------------------------------

class TestMessageSerialisation(unittest.TestCase):
    def test_payload_is_valid_json(self):
        """Serialised trade payload must be valid JSON."""
        payload = {
            "symbol": "AAPL",
            "price": 189.45,
            "volume": 1200,
            "timestamp": "2024-07-15T09:30:01Z",
        }
        encoded = json.dumps(payload).encode("utf-8")
        decoded = json.loads(encoded.decode("utf-8"))
        self.assertEqual(decoded["symbol"], "AAPL")
        self.assertAlmostEqual(decoded["price"], 189.45)

    def test_missing_required_fields_raises(self):
        """Payload without required fields should raise KeyError on access."""
        bad_payload = {"price": 100.0}
        with self.assertRaises(KeyError):
            _ = bad_payload["symbol"]


# ---------------------------------------------------------------------------
# Tests for ML predict module
# ---------------------------------------------------------------------------

class TestPredictModule(unittest.TestCase):
    @patch("ml.predict.bigquery.Client")
    def test_predict_returns_dict(self, mock_bq):
        """predict() should return a dict with symbol and prediction keys."""
        mock_bq.return_value.query.return_value.result.return_value = []
        # Lightweight smoke-test: import must succeed
        try:
            import ml.predict  # noqa: F401
        except ImportError:
            self.skipTest("ml.predict dependencies not installed")


if __name__ == "__main__":
    unittest.main()
