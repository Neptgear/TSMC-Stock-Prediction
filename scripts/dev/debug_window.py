import pprint
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_fetch import get_ohlcv, get_event_countdowns
from train_transformer import standardize_ohlcv_columns
from ta_features import compute_features, compute_indicators_only, get_tft_feature_columns, get_default_feature_columns


def main() -> None:
    ticker = "2330.TW"
    start = "2010-01-01"
    target = "next_close"
    horizon = 10
    window_size = 180

    df = get_ohlcv(ticker_yt=ticker, start=start, end=None, auto_adjust=True)
    df = standardize_ohlcv_columns(df)
    print("raw_rows", len(df), "index_type", type(df.index))

    events_df = get_event_countdowns(ticker, df.index)
    print("events_rows", len(events_df), "event_cols", list(events_df.columns))

    feat_df_base = compute_features(df, target=target, horizon=horizon, include_extra=True, events_df=events_df)
    print("feat_df_base rows", len(feat_df_base), "cols", len(feat_df_base.columns))

    charts_df = compute_indicators_only(df, include_extra=True)
    print("charts_df rows", len(charts_df), "cols", len(charts_df.columns))

    feature_cols_tft = get_tft_feature_columns(feat_df_base)
    print("tft feature count", len(feature_cols_tft))

    # Emulate the drop in run_training
    ta_cols = [c for c in feature_cols_tft if not (isinstance(c, str) and c.startswith("f_"))]
    print("ta_cols", len(ta_cols))
    feat_df = feat_df_base.dropna(subset=ta_cols + ["target"])
    print("after dropna rows_avail", len(feat_df))

    eff_window_req = int(window_size)
    rows_avail = len(feat_df)
    eff_window = eff_window_req if rows_avail > eff_window_req else max(5, rows_avail - 1)
    num_samples_est = max(0, rows_avail - eff_window + 1)
    print("eff_window_req", eff_window_req, "eff_window", eff_window, "num_samples_est", num_samples_est)

    # Inspect per-column NaN counts for TA columns
    na_counts = feat_df_base[ta_cols + ["target"]].isna().sum()
    print("NaNs per TA/target:")
    pprint.pprint(na_counts.to_dict())

    if "is_holiday_eve" in feat_df_base.columns:
        print("is_holiday_eve sample:", feat_df_base["is_holiday_eve"].head(10).tolist())
        print("is_holiday_eve unique (incl NaN):", feat_df_base["is_holiday_eve"].drop_duplicates().tolist())


if __name__ == "__main__":
    main()
