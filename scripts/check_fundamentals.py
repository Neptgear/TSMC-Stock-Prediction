import sys
import pathlib
import pandas as pd

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_fetch import get_ohlcv, get_fundamentals_timeseries


def main(ticker: str) -> None:
    print(f"Ticker: {ticker}")
    try:
        df = get_ohlcv(ticker_yt=ticker, start="2015-01-01", end=None, auto_adjust=True)
        align_index = df.index
    except Exception as ex:
        print(f"OHLCV fetch failed: {ex}")
        align_index = None

    fdf = get_fundamentals_timeseries(ticker, align_index=align_index)
    print(f"Fundamentals shape: {fdf.shape}")
    print(f"Columns ({len(fdf.columns)}): {list(fdf.columns)[:20]}{'...' if len(fdf.columns) > 20 else ''}")
    # Show last 3 non-null rows for a few key columns
    for key in [
        "f_rev",
        "f_net_income",
        "f_gross_margin",
        "f_op_margin",
        "f_current_ratio",
        "f_debt_to_equity",
    ]:
        if key in fdf.columns:
            s = fdf[key].dropna().tail(3)
            print(f"Sample {key}:\n{s}")
    print()


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["2330.TW", "AAPL"]
    for t in tickers:
        main(t)
