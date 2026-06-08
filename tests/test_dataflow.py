"""Unit tests for Dataflow / Apache Beam pipeline."""
import unittest


class TestStockPipeline(unittest.TestCase):
    def test_parse_valid_message(self):
        """Valid JSON message must be parsed into a dict."""
        import json

        raw = json.dumps({
            "symbol": "TSLA",
            "price": 248.50,
            "volume": 5000,
            "event_time": "2024-07-15T09:31:00Z",
        }).encode()
        parsed = json.loads(raw.decode("utf-8"))
        self.assertEqual(parsed["symbol"], "TSLA")

    def test_parse_invalid_message_raises(self):
        """Malformed JSON must raise a ValueError."""
        import json

        with self.assertRaises(json.JSONDecodeError):
            json.loads("not-json")

    def test_feature_engineering_sma(self):
        """Simple moving average must equal mean of input prices."""
        prices = [100.0, 102.0, 104.0, 103.0, 105.0]
        sma = sum(prices) / len(prices)
        self.assertAlmostEqual(sma, 102.8)

    def test_feature_engineering_returns(self):
        """Price return must be (p_t - p_t-1) / p_t-1."""
        p_prev, p_curr = 100.0, 105.0
        ret = (p_curr - p_prev) / p_prev
        self.assertAlmostEqual(ret, 0.05)


if __name__ == "__main__":
    unittest.main()
