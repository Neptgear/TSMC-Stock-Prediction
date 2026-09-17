import unittest
from unittest.mock import patch

import pandas as pd

import data_fetch


class _FakeYFinance:
    @staticmethod
    def download(*args, **kwargs):
        columns = pd.MultiIndex.from_tuples(
            [
                ("Open", "2330.TW"),
                ("High", "2330.TW"),
                ("Low", "2330.TW"),
                ("Close", "2330.TW"),
                ("Volume", "2330.TW"),
            ],
            names=["Price", "Ticker"],
        )
        return pd.DataFrame(
            [[100.0, 102.0, 99.0, 101.0, 1000]],
            index=pd.to_datetime(["2025-01-02"]),
            columns=columns,
        )


class DataFetchTests(unittest.TestCase):
    def test_yfinance_field_first_multiindex_is_normalized_and_audited(self):
        with patch.object(data_fetch, "yf", _FakeYFinance()):
            frame = data_fetch.get_ohlcv(
                ticker_yt="2330.TW",
                start="2025-01-01",
                end="2025-01-02",
                provider="yahoo",
            )

        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(frame.attrs["data_quality"]["source"], "yahoo_finance")
        self.assertEqual(frame.attrs["data_quality"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()

