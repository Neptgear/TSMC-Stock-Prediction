import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import create_app


class AppSmokeTests(unittest.TestCase):
    def _market_frame(self):
        index = pd.bdate_range("2025-01-01", periods=80)
        close = np.linspace(900.0, 980.0, len(index))
        frame = pd.DataFrame(
            {
                "Open": close - 2.0,
                "High": close + 4.0,
                "Low": close - 5.0,
                "Close": close,
                "Volume": np.full(len(index), 20_000_000.0),
            },
            index=index,
        )
        return frame

    @patch("app.list_saved_runs", return_value=[])
    @patch("app.get_default_feature_columns", return_value=["Close", "Volume"])
    @patch("app.compute_features")
    @patch("app.compute_indicators_only")
    @patch("app.get_event_countdowns")
    @patch("app.get_ohlcv")
    def test_home_page_renders_without_external_network(
        self,
        get_ohlcv,
        get_event_countdowns,
        compute_indicators_only,
        compute_features,
        _get_default_feature_columns,
        _list_saved_runs,
    ):
        frame = self._market_frame()
        get_ohlcv.return_value = frame
        get_event_countdowns.return_value = pd.DataFrame(index=frame.index)
        compute_indicators_only.return_value = frame
        compute_features.return_value = frame.assign(target=frame["Close"].shift(-1))

        flask_app = create_app()
        flask_app.config.update(TESTING=True)
        response = flask_app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"TA Stock Transformer", response.data)
        self.assertIn(b"2330.TW", response.data)


if __name__ == "__main__":
    unittest.main()

