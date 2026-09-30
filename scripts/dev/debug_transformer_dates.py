import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_transformer import TrainConfig, run_training, standardize_ohlcv_columns
from data_fetch import get_ohlcv, get_fundamentals_timeseries
from ta_features import compute_features, get_default_feature_columns


def main() -> None:
    cfg = TrainConfig(
        ticker="2330.TW",
        start="2020-01-01",
        end=None,
        target="next_close",
        horizon=1,
        window_size=60,
        epochs=1,
        use_fundamentals=True,
        split_date="2024-12-31",
        model_type="transformer",
    )
    res = run_training(cfg)
    dates = res["test_dates"]
    print("num_test", len(dates))
    # Show a slice around April–May 2025
    for d, y in zip(dates, res["y_true"]):
        if "2025-03" <= d <= "2025-06":
            print(d, y)

    # Inspect raw df index around that period to see if dates exist.
    df = get_ohlcv(ticker_yt="2330.TW", start="2020-01-01", end=None, auto_adjust=True)
    df = standardize_ohlcv_columns(df)
    print("raw index around 2025-03 to 2025-06:")
    for d in df.index:
        s = str(d.date())
        if "2025-03" <= s <= "2025-06":
            print("df:", s)

    # Reproduce feature pipeline to see which rows are dropped for Transformer.
    feat_df_base = compute_features(df, target="next_close", horizon=1, include_extra=True)
    feat_df = feat_df_base.copy()
    # Join fundamentals similarly to run_training (no events for simplicity).
    try:
        fund_df = get_fundamentals_timeseries("2330.TW", align_index=df.index)
        import pandas as _pd
        if isinstance(fund_df, _pd.DataFrame) and not fund_df.empty:
            feat_df = feat_df.join(fund_df, how="left")
            fcols = [c for c in feat_df.columns if isinstance(c, str) and c.startswith("f_")]
            good_f_cols = [c for c in fcols if feat_df[c].notna().sum() >= 4]
            drop_f_cols = [c for c in fcols if c not in good_f_cols]
            if drop_f_cols:
                feat_df = feat_df.drop(columns=drop_f_cols)
    except Exception:
        pass

    feature_cols = get_default_feature_columns(feat_df)
    fund_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith("f_")]
    ta_cols = [c for c in feature_cols if c not in fund_cols]
    feat_df_dropped = feat_df.dropna(subset=ta_cols + ["target"])
    print("feat_df rows", len(feat_df), "after drop", len(feat_df_dropped))
    print("dates kept 2025-03..2025-06:")
    for d in feat_df_dropped.index:
        s = str(d.date())
        if "2025-03" <= s <= "2025-06":
            print("feat_df_dropped:", s)

    print("rows with NaNs in TA cols between 2025-04-03 and 2025-05-08:")
    for d in feat_df.index:
        s = str(d.date())
        if "2025-04-03" <= s <= "2025-05-08":
            row = feat_df.loc[d, ta_cols]
            bad = row[row.isna()]
            if not bad.empty:
                print("date", s, "NaN cols:", list(bad.index))


if __name__ == "__main__":
    main()
