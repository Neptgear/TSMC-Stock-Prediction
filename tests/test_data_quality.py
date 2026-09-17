import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_quality import audit_ohlcv, audit_saved_run


class OhlcvAuditTests(unittest.TestCase):
    def test_sorts_and_records_duplicate_dates(self):
        frame = pd.DataFrame(
            {
                "Open": [101, 100, 102],
                "High": [103, 102, 104],
                "Low": [100, 99, 101],
                "Close": [102, 101, 103],
                "Volume": [1100, 1000, 1200],
            },
            index=pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-02"]),
        )
        clean, report = audit_ohlcv(frame)
        self.assertTrue(clean.index.is_monotonic_increasing)
        self.assertTrue(clean.index.is_unique)
        self.assertEqual(report["duplicate_dates_removed"], 1)
        self.assertEqual(len(clean), 2)

    def test_rejects_impossible_price_bar(self):
        frame = pd.DataFrame(
            {"Open": [100], "High": [99], "Low": [98], "Close": [101], "Volume": [1000]},
            index=pd.to_datetime(["2025-01-01"]),
        )
        with self.assertRaisesRegex(ValueError, "price/volume constraints"):
            audit_ohlcv(frame)

    def test_allows_subcent_adjustment_rounding(self):
        frame = pd.DataFrame(
            {
                "Open": [475.0],
                "High": [478.64790],
                "Low": [473.0],
                "Close": [478.64791],
                "Volume": [1000],
            },
            index=pd.to_datetime(["2020-12-31"]),
        )
        clean, report = audit_ohlcv(frame)
        self.assertEqual(len(clean), 1)
        self.assertEqual(report["invalid_market_rows"], 0)


class SavedRunAuditTests(unittest.TestCase):
    def _write_json(self, path: Path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_recalculates_metrics_and_accepts_consistent_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = np.array([100.0, 102.0, 104.0])
            predicted = np.array([101.0, 101.0, 105.0])
            np.save(root / "y_true.npy", actual)
            np.save(root / "y_pred.npy", predicted)
            self._write_json(root / "test_dates.json", ["2025-01-01", "2025-01-02", "2025-01-03"])
            diff = actual - predicted
            self._write_json(
                root / "metrics.json",
                {"mse": float(np.mean(diff**2)), "rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))},
            )
            self._write_json(root / "config.json", {"train_ratio": 0.8})
            self._write_json(
                root / "split_info.json",
                {"split_rule": "ratio_samples_0.80", "test_start": "2025-01-01", "test_end": "2025-01-03"},
            )
            report = audit_saved_run(root)
            self.assertTrue(report["passed"], report)

    def test_flags_split_ratio_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.array([1.0, 2.0])
            np.save(root / "y_true.npy", values)
            np.save(root / "y_pred.npy", values)
            self._write_json(root / "test_dates.json", ["2025-01-01", "2025-01-02"])
            self._write_json(root / "metrics.json", {"mse": 0.0, "rmse": 0.0, "mae": 0.0})
            self._write_json(root / "config.json", {"train_ratio": 0.6})
            self._write_json(root / "split_info.json", {"split_rule": "ratio_samples_0.80"})
            report = audit_saved_run(root)
            self.assertFalse(report["passed"])
            self.assertTrue(any("conflicts" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()

