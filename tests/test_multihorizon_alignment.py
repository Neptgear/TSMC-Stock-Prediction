import unittest

import numpy as np
import pandas as pd

from train_transformer import (
    TrainConfig,
    _configured_future_dates,
    _future_business_dates,
    _future_known_matrix,
    build_seq2seq_tensors,
)


class MultiHorizonAlignmentTests(unittest.TestCase):
    def test_first_target_is_first_row_after_encoder_anchor(self):
        dates = pd.bdate_range("2025-01-01", periods=10)
        close = np.arange(100.0, 110.0, dtype=np.float32)
        frame = pd.DataFrame(
            {
                "Close": close,
                "day_of_week": dates.dayofweek.astype(float),
                "target": close,
            },
            index=dates,
        )

        _, _, _, _, targets, bases, anchors, target_dates = build_seq2seq_tensors(
            frame,
            raw_close=frame["Close"],
            observed_cols=["Close"],
            known_cols=["day_of_week"],
            static_cols=[],
            target_col="target",
            horizon=4,
            window_size=3,
        )

        np.testing.assert_array_equal(targets[0], close[3:7])
        self.assertEqual(bases[0], close[2])
        self.assertEqual(anchors[0], dates[2])
        self.assertEqual(target_dates[0], dates[6])

    def test_asof_dates_start_after_cutoff_without_price_data(self):
        dates = _future_business_dates(pd.Timestamp("2026-09-11"), 4)
        self.assertEqual(
            [str(value.date()) for value in dates],
            ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17"],
        )
        matrix = _future_known_matrix(dates, ["day_of_week", "is_month_end"])
        np.testing.assert_array_equal(matrix[:, 0], np.array([0, 1, 2, 3], dtype=np.float32))

    def test_known_exchange_schedule_can_skip_a_weekday_holiday(self):
        cfg = TrainConfig(
            horizon=4,
            forecast_dates=("2026-06-18", "2026-06-22", "2026-06-23", "2026-06-24"),
        )
        dates = _configured_future_dates(pd.Timestamp("2026-06-17"), cfg)
        self.assertEqual(
            [str(value.date()) for value in dates],
            ["2026-06-18", "2026-06-22", "2026-06-23", "2026-06-24"],
        )


if __name__ == "__main__":
    unittest.main()

