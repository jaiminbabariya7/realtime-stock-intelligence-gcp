"""Unit tests for the ingestion module."""
import importlib
import importlib.util
import json
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "ingestion"))
sys.path.insert(0, os.path.join(ROOT, "dataflow"))

# symbol_config reads GCP_PROJECT_ID at import time
os.environ.setdefault("GCP_PROJECT_ID", "test-project")


# ---------------------------------------------------------------------------
# Tests for symbol_config
# ---------------------------------------------------------------------------

class TestSymbolConfig(unittest.TestCase):
    def test_symbols_not_empty(self):
        """SYMBOLS list must contain at least one ticker."""
        from symbol_config import SYMBOLS
        self.assertIsInstance(SYMBOLS, list)
        self.assertGreater(len(SYMBOLS), 0)

    def test_symbols_are_strings(self):
        """All symbols must be non-empty uppercase strings."""
        from symbol_config import SYMBOLS
        for s in SYMBOLS:
            self.assertIsInstance(s, str)
            self.assertTrue(s.isupper(), f"Symbol {s!r} is not uppercase")
            self.assertGreater(len(s), 0)

    def test_sma_windows_ordered(self):
        """Short moving-average window must be shorter than the long one."""
        from symbol_config import SMA_LONG, SMA_SHORT
        self.assertLess(SMA_SHORT, SMA_LONG)


# ---------------------------------------------------------------------------
# Tests for message serialisation in alpaca_stream
# ---------------------------------------------------------------------------

def _import_alpaca_stream():
    """Import alpaca_stream with the Alpaca SDK and Pub/Sub client stubbed out."""
    alpaca_live = types.ModuleType("alpaca.data.live")
    alpaca_live.StockDataStream = MagicMock()
    stubs = {
        "alpaca": types.ModuleType("alpaca"),
        "alpaca.data": types.ModuleType("alpaca.data"),
        "alpaca.data.live": alpaca_live,
        "google.cloud.pubsub_v1": MagicMock(),
    }
    env = {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test"}
    with patch.dict(sys.modules, stubs), patch.dict(os.environ, env):
        sys.modules.pop("alpaca_stream", None)
        return importlib.import_module("alpaca_stream")


class TestMessageSerialisation(unittest.TestCase):
    def setUp(self):
        self.mod = _import_alpaca_stream()
        self.trade = types.SimpleNamespace(
            symbol="AAPL", price=189.45, size=1200, id=42, exchange="V",
            timestamp=datetime(2024, 7, 15, 9, 30, 1, tzinfo=timezone.utc))

    def test_payload_fields(self):
        """Payload must carry symbol, price, volume and an ISO timestamp."""
        payload = json.loads(self.mod._build_payload(self.trade))
        self.assertEqual(payload["symbol"], "AAPL")
        self.assertAlmostEqual(payload["price"], 189.45)
        self.assertEqual(payload["volume"], 1200)
        self.assertEqual(payload["trade_id"], "42")
        self.assertEqual(payload["timestamp"], "2024-07-15T09:30:01+00:00")

    def test_payload_accepted_by_pipeline_parser(self):
        """What ingestion publishes must pass the Dataflow parser unchanged."""
        from features import parse_tick
        record = parse_tick(self.mod._build_payload(self.trade))
        self.assertIsNotNone(record)
        self.assertEqual(record["symbol"], "AAPL")


# ---------------------------------------------------------------------------
# Tests for ML predict module
# ---------------------------------------------------------------------------

class TestPredictModule(unittest.TestCase):
    def test_predict_module_imports(self):
        """Smoke test: ml.predict imports when the ML dependencies are installed."""
        if importlib.util.find_spec("pytorch_forecasting") is None:
            self.skipTest("ML dependencies (ml/requirements.txt) not installed")
        sys.path.insert(0, ROOT)
        importlib.import_module("ml.predict")


if __name__ == "__main__":
    unittest.main()
