import unittest

import numpy as np
import pandas as pd

from scripts.asof_backtest import _price_baselines
from scripts.rolling_model_backtest import _summarize
from scripts.rolling_price_baselines import evaluate_baselines


class PriceObjectiveTests(unittest.TestCase):
    def test_baselines_use_only_cutoff_history(self):
        frame = pd.DataFrame({"Close": np.arange(1.0, 61.0)})
        dates = [
            {"date": "2026-09-14"},
            {"date": "2026-09-15"},
            {"date": "2026-09-16"},
            {"date": "2026-09-17"},
        ]
        baselines = _price_baselines(frame, dates, cutoff_close=60.0)

        self.assertEqual(
            [row["value"] for row in baselines["naive_last_close"]["forecast"]],
            [60.0] * 4,
        )
        self.assertEqual(
            [row["value"] for row in baselines["sma_5"]["forecast"]],
            [58.0] * 4,
        )
        self.assertEqual(
            [round(row["value"], 6) for row in baselines["linear_trend_60"]["forecast"]],
            [61.0, 62.0, 63.0, 64.0],
        )

    def test_rolling_origin_uses_non_overlapping_future_blocks(self):
        index = pd.bdate_range("2025-01-01", periods=140)
        frame = pd.DataFrame({"Close": np.arange(1.0, 141.0)}, index=index)
        report = evaluate_baselines(frame, periods=5, horizon=4)

        self.assertEqual(len(report["periods"]), 5)
        self.assertEqual(report["ranking_by_mae"][0], "linear_trend_60")
        self.assertAlmostEqual(report["summary"]["linear_trend_60"]["mae"], 0.0, places=8)
        test_dates = [
            date
            for period in report["periods"]
            for date in pd.bdate_range(period["test_start"], period["test_end"])
        ]
        self.assertEqual(len(test_dates), len(set(test_dates)))

    def test_model_summary_requires_every_seed_below_baseline(self):
        completed = []
        for seed in (11, 42, 97):
            completed.append(
                {
                    "status": "ok",
                    "model": "transformer",
                    "seed": seed,
                    "cutoff_close": 10.0,
                    "actual": [12.0, 8.0],
                    "errors": [1.0, -1.0],
                }
            )
        baseline = {"summary": {"naive_last_close": {"mae": 2.0}}}
        threshold, summary = _summarize(
            completed, baseline, models=["transformer"], seeds=[11, 42, 97]
        )
        self.assertEqual(threshold, 2.0)
        self.assertTrue(summary["transformer"]["stable_below_naive_threshold"])
        self.assertTrue(summary["transformer"]["all_seeds_statistically_clear"])


if __name__ == "__main__":
    unittest.main()

