import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data_fetch import get_ohlcv, _fetch_alpha_vantage_ohlcv, _fetch_yf_series, _fetch_cboe_vix_series


def summarize_diff(df_y: pd.DataFrame, df_a: pd.DataFrame, cols=("Open","High","Low","Close")):
    idx = df_y.index.intersection(df_a.index)
    y = df_y.loc[idx, list(c for c in cols if c in df_y.columns)]
    a = df_a.loc[idx, list(c for c in cols if c in df_a.columns)]
    out = {}
    for c in set(y.columns).intersection(a.columns):
        dy = (y[c] - a[c]).abs()
        out[c] = {
            "n": int(dy.notna().sum()),
            "mad": float(dy.mean(skipna=True) or 0.0),
            "max": float(dy.max(skipna=True) or 0.0),
            "corr": float(y[c].corr(a[c])) if y[c].notna().any() and a[c].notna().any() else float("nan"),
            "diff_days": int((dy > 1e-6).sum()),
        }
    return idx, out


def main():
    ticker = os.getenv("TICKER", "2330.TW")
    start = os.getenv("START", "2015-01-01")

    print(f"Comparing sources for {ticker} from {start}...")
    try:
        df_y = get_ohlcv(ticker, start=start, provider="yahoo")
        df_a = get_ohlcv(ticker, start=start, provider="alpha")
    except Exception:
        # Try ADR TSM for both to enable a like-for-like comparison if TWSE is unavailable on AV
        print("Alpha Vantage failed for the given symbol; retrying comparison with ADR 'TSM'.")
        ticker = "TSM"
        df_y = get_ohlcv(ticker, start=start, provider="yahoo")
        df_a = get_ohlcv(ticker, start=start, provider="alpha")
    idx, stats = summarize_diff(df_y, df_a)
    print(f"Rows (Yahoo): {len(df_y)}, Rows (Alpha): {len(df_a)}, Intersection: {len(idx)}")
    for k, v in stats.items():
        print(f" {k}: n={v['n']} mad={v['mad']:.6f} max={v['max']:.6f} corr={v['corr']:.6f} diff_days={v['diff_days']}")

    # VIX comparison: Yahoo vs CBOE
    vix_y = _fetch_yf_series("^VIX", start)
    vix_c = _fetch_cboe_vix_series("vix")
    if vix_y is not None and vix_c is not None:
        # Normalize tz to naive
        if getattr(vix_y.index, "tz", None) is not None:
            vix_y = pd.Series(vix_y.values, index=vix_y.index.tz_localize(None), name=vix_y.name)
        if getattr(vix_c.index, "tz", None) is not None:
            vix_c = pd.Series(vix_c.values, index=vix_c.index.tz_localize(None), name=vix_c.name)
        i2 = vix_y.index.intersection(vix_c.index)
        dy = (vix_y.loc[i2] - vix_c.loc[i2]).abs()
        print(f"VIX: intersection={len(i2)}, mad={dy.mean():.6f}, max={dy.max():.6f}")
    else:
        print("VIX comparison skipped (missing series)")


if __name__ == "__main__":
    main()
