import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from data_fetch import get_ohlcv, get_fundamentals_timeseries, get_event_countdowns
from ta_features import compute_features, get_default_feature_columns
from train_transformer import (
    TimeSeriesTransformer,
    standardize_ohlcv_columns,
)


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + eps)
    r2 = float(1 - ss_res / ss_tot)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def list_runs(root: str = "runs") -> list:
    p = Path(root)
    if not p.exists():
        return []
    items = []
    for d in sorted(p.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = {"name": d.name, "path": str(d)}
        try:
            cfg_path = d / "config.json"
            if cfg_path.exists():
                meta.update(json.load(open(cfg_path, "r", encoding="utf-8")))
            y_true = d / "y_true.npy"
            y_pred = d / "y_pred.npy"
            if y_true.exists() and y_pred.exists():
                yt = np.load(y_true)
                yp = np.load(y_pred)
                meta.update(compute_metrics_np(yt, yp))
        except Exception:
            pass
        items.append(meta)
    return items


def select_best_run(runs: list) -> Optional[dict]:
    scored = [r for r in runs if isinstance(r.get("rmse"), (int, float))]
    if not scored:
        return None
    return sorted(scored, key=lambda r: r["rmse"])[0]


def _load_scaler_state(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pickle as _pickle

    st = _pickle.load(open(path, "rb"))
    mean = np.array(st["mean_"]).astype(np.float32)
    std = np.array(st["std_"]).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def predict_with_run(
    run_dir: str,
    ticker: str = "2330.TW",
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> Dict:
    d = Path(run_dir)
    cfg = json.load(open(d / "config.json", "r", encoding="utf-8"))
    model_state = torch.load(d / "model.pt", map_location="cpu")
    mean, std = _load_scaler_state(d / "x_scaler.pkl")
    y_scaler = None
    ys = d / "y_scaler.json"
    if ys.exists():
        y_scaler = json.load(open(ys, "r", encoding="utf-8"))

    h = int(cfg.get("horizon", 1))
    w = int(cfg.get("window_size", 30))
    target = cfg.get("target", "next_close")
    residualize = bool(cfg.get("residualize", True))
    residual_base = cfg.get("residual_base", "close")

    # Fetch and prepare data
    df = get_ohlcv(ticker_yt=ticker, start=start, end=end, auto_adjust=True)
    df = standardize_ohlcv_columns(df)
    events_df = get_event_countdowns(ticker, df.index)
    feat_df = compute_features(df, target=target, horizon=h, include_extra=True, events_df=events_df)
    # Join fundamentals to mirror training
    try:
        fund_df = get_fundamentals_timeseries(ticker, align_index=df.index)
        import pandas as _pd
        if isinstance(fund_df, _pd.DataFrame) and not fund_df.empty:
            overlap = [c for c in fund_df.columns if c in feat_df.columns]
            if overlap:
                fund_df = fund_df.drop(columns=overlap)
            feat_df = feat_df.join(fund_df, how="left")
            # Remove weak fundamental columns with no data
            fcols = [c for c in feat_df.columns if isinstance(c, str) and c.startswith("f_")]
            good_f = [c for c in fcols if feat_df[c].notna().sum() >= 4]
            drop_f = [c for c in fcols if c not in good_f]
            if drop_f:
                feat_df = feat_df.drop(columns=drop_f)
    except Exception:
        pass
    # Use training-time feature columns when present for strict consistency
    cfg_feature_cols = cfg.get("feature_cols") if isinstance(cfg, dict) else None
    if isinstance(cfg_feature_cols, list) and len(cfg_feature_cols) > 0:
        feature_cols = [c for c in cfg_feature_cols if c in feat_df.columns]
        # If none of the saved columns exist (code drift), fallback gently
        if not feature_cols:
            feature_cols = get_default_feature_columns(feat_df)
    else:
        feature_cols = get_default_feature_columns(feat_df)
    # Drop rows only where selected features/target are missing
    feat_df = feat_df.dropna(subset=feature_cols + ["target"])
    values = feat_df[feature_cols].values.astype(np.float32)
    targets = feat_df["target"].values.astype(np.float32)
    if len(values) <= w:
        raise RuntimeError("Not enough rows to build the last window.")

    # Build windows (include the last sample).
    xs, ys_list = [], []
    for end_idx in range(w, len(values) + 1):
        xs.append(values[end_idx - w:end_idx])
        ys_list.append(targets[end_idx - 1])
    X_all = np.stack(xs, axis=0).astype(np.float32)
    y_all = np.array(ys_list).astype(np.float32)

    # Scale features
    X_all = (X_all - mean) / std

    # Build model and run inference
    model = TimeSeriesTransformer(
        input_size=X_all.shape[-1],
        d_model=int(cfg.get("d_model", 128)),
        nhead=int(cfg.get("nhead", 4)),
        num_layers=int(cfg.get("num_layers", 3)),
        dim_feedforward=int(cfg.get("dim_feedforward", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(model_state)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(X_all)).cpu().numpy().reshape(-1)

    # Inverse-scale model output only. Ground truth `y_all` is already in
    # original units produced by `compute_features` and must not be scaled.
    if y_scaler is not None:
        y_pred_eff = y_pred_scaled * float(y_scaler["std"]) + float(y_scaler["mean"])
    else:
        y_pred_eff = y_pred_scaled
    # Ground truth in effective space
    y_true_eff = y_all

    # Reconstruct price if residualized
    if target == "next_close" and residualize:
        base_col = "Close" if residual_base == "close" else ("ma20" if "ma20" in feat_df.columns else "Close")
        base_series = feat_df[base_col].values.astype(np.float32)
        base_aligned = base_series[w - 1: w - 1 + len(y_true_eff)]
        # Reconstruct predicted next_close from residual + baseline
        y_pred = y_pred_eff + base_aligned
        # Ground truth is already next_close; do NOT add baseline again
        y_true = y_true_eff
    else:
        y_pred = y_pred_eff
        y_true = y_true_eff

    # Dates for plotting: compute target dates using the RAW df index so that
    # the last label reflects the next trading day even if feat_df dropped it.
    import pandas as _pd
    anchors = feat_df.index[w - 1 : w - 1 + len(y_true)]
    apos = _pd.Index(df.index).get_indexer(anchors)
    tpos = apos + int(h)
    tpos = np.clip(tpos, 0, len(df.index) - 1)
    tdates = _pd.Index(df.index)[tpos]
    test_dates = [str(d.date()) if hasattr(d, 'date') else str(d) for d in tdates]

    # Future pred at End+h using last window
    future_point = None
    actual_target = None
    future_point_error = None
    try:
        last_window = values[-w:].astype(np.float32)
        last_scaled = (last_window - mean) / std
        with torch.no_grad():
            y_last = model(torch.from_numpy(last_scaled[None, ...])).cpu().numpy().reshape(-1)[0]
        if y_scaler is not None:
            y_last_eff = y_last * float(y_scaler["std"]) + float(y_scaler["mean"])
        else:
            y_last_eff = y_last
        if target == "next_close" and residualize:
            base_col = "Close" if residual_base == "close" else ("ma20" if "ma20" in feat_df.columns else "Close")
            base_last = float(feat_df[base_col].iloc[-1])
        else:
            base_last = 0.0
        fval = y_last_eff + base_last if base_last != 0.0 else y_last_eff
        end_idx = df.index[-1]
        # Prefer the next trading date(s) from Yahoo if available; fallback to BDay
        try:
            df_after = get_ohlcv(ticker_yt=ticker, start=str(getattr(end_idx, 'date', lambda: end_idx)()), end=None, auto_adjust=True)
            df_after = standardize_ohlcv_columns(df_after)
            next_trading = df_after.index[df_after.index > end_idx]
            if len(next_trading) >= h:
                fdate = next_trading[h - 1]
            elif len(next_trading) > 0:
                fdate = next_trading[-1] + _pd.tseries.offsets.BDay(h - len(next_trading))
            else:
                fdate = end_idx + _pd.tseries.offsets.BDay(h)
        except Exception:
            fdate = end_idx + _pd.tseries.offsets.BDay(h)
        future_point = {"date": str(getattr(fdate, 'date', lambda: fdate)()), "value": float(fval)}

        # actual at exact target date only
        tgt_str = str(getattr(fdate, 'date', lambda: fdate)())
        df_t = get_ohlcv(ticker_yt=ticker, start=start, end=tgt_str, auto_adjust=True)
        df_t = standardize_ohlcv_columns(df_t)
        if fdate in df_t.index:
            actual_target = {"date": tgt_str, "value": float(df_t["Close"].loc[fdate])}
    except Exception:
        pass

    metrics = compute_metrics_np(y_true, y_pred)
    return {
        "metrics": metrics,
        "feature_cols": feature_cols,
        "X_shape": list(X_all.shape),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "test_dates": test_dates,
        "future_point": future_point,
        "actual_target": actual_target,
    }


def load_run_results(run_dir: str) -> Dict:
    """Load saved predictions from a training run directory without recomputing.

    Returns a structure compatible with the UI's expectations in app.py/templates.
    Prefers persisted test_dates/future/actual artifacts if available.
    """
    d = Path(run_dir)
    cfg = json.load(open(d / "config.json", "r", encoding="utf-8"))
    # Load saved arrays
    y_true = np.load(d / "y_true.npy") if (d / "y_true.npy").exists() else np.array([])
    y_pred = np.load(d / "y_pred.npy") if (d / "y_pred.npy").exists() else np.array([])
    # Optional stored metrics / split info
    metrics_file = d / "metrics.json"
    metrics_train_file = d / "metrics_train.json"
    split_info_file = d / "split_info.json"
    diagnostics_file = d / "diagnostics.json"
    metrics_saved = None
    metrics_train_saved = None
    split_info = None
    diagnostics = None
    try:
        if metrics_file.exists():
            metrics_saved = json.load(open(metrics_file, "r", encoding="utf-8"))
            # Support both flat metrics dict and {"test":{...},"train":{...}}
            if isinstance(metrics_saved, dict):
                if "test" in metrics_saved:
                    metrics_train_saved = metrics_saved.get("train")
                    metrics_saved = metrics_saved.get("test")
                elif "metrics_train" in metrics_saved:
                    metrics_train_saved = metrics_saved.get("metrics_train")
            else:
                metrics_saved = None
    except Exception:
        metrics_saved = None
    try:
        if metrics_train_file.exists():
            metrics_train_saved = json.load(open(metrics_train_file, "r", encoding="utf-8"))
    except Exception:
        pass
    try:
        if split_info_file.exists():
            split_info = json.load(open(split_info_file, "r", encoding="utf-8"))
    except Exception:
        split_info = None
    try:
        if diagnostics_file.exists():
            diagnostics = json.load(open(diagnostics_file, "r", encoding="utf-8"))
    except Exception:
        diagnostics = None

    # Try to load saved test dates and optional points
    test_dates = None
    try:
        td_path = d / "test_dates.json"
        if td_path.exists():
            test_dates = json.load(open(td_path, "r", encoding="utf-8"))
    except Exception:
        test_dates = None

    future_point = None
    actual_target = None
    try:
        fp = d / "future_point.json"
        if fp.exists():
            future_point = json.load(open(fp, "r", encoding="utf-8"))
        ap = d / "actual_target.json"
        if ap.exists():
            actual_target = json.load(open(ap, "r", encoding="utf-8"))
        fe = d / "future_point_error.json"
        if fe.exists():
            try:
                future_point_error = json.load(open(fe, "r", encoding="utf-8")).get("error")
            except Exception:
                future_point_error = None
    except Exception:
        pass

    f1_by_horizon = None
    rolling_f1 = None
    try:
        f1_path = d / "f1_by_horizon.json"
        if f1_path.exists():
            f1_by_horizon = json.load(open(f1_path, "r", encoding="utf-8"))
    except Exception:
        f1_by_horizon = None
    try:
        roll_path = d / "rolling_f1.json"
        if roll_path.exists():
            rolling_f1 = json.load(open(roll_path, "r", encoding="utf-8"))
    except Exception:
        rolling_f1 = None

    # Fallback: attempt to parse runs/predictions.csv for this run_id to recover dates
    run_id = d.name
    if (test_dates is None or len(test_dates) == 0) and (Path("runs") / "predictions.csv").exists():
        try:
            csv_path = Path("runs") / "predictions.csv"
            lines = open(csv_path, "r", encoding="utf-8").read().splitlines()
            dates_csv, true_csv, pred_csv = [], [], []
            in_block = False
            for line in lines:
                if line.startswith("# RUN "):
                    # entering a new block; toggle based on run_id match
                    in_block = (f"# RUN {run_id} " in line) or line.strip().startswith(f"# RUN {run_id}\n")
                    continue
                if not in_block:
                    continue
                if not line or line.startswith("#"):
                    continue
                # Expect CSV rows or header; skip header
                if line.lower().startswith("date,true,pred"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    dates_csv.append(parts[0])
                    try:
                        true_csv.append(float(parts[1]) if parts[1] != '' else None)
                    except Exception:
                        true_csv.append(None)
                    try:
                        pred_csv.append(float(parts[2]) if parts[2] != '' else None)
                    except Exception:
                        pred_csv.append(None)
            # If we collected rows and lengths match saved arrays (or saved arrays empty), use them
            if dates_csv and ((len(y_true) == 0 and len(y_pred) == 0) or (len(dates_csv) == len(y_true) == len(y_pred))):
                test_dates = dates_csv
                if len(y_true) == 0:
                    y_true = np.array([v if v is not None else np.nan for v in true_csv], dtype=float)
                if len(y_pred) == 0:
                    y_pred = np.array([v if v is not None else np.nan for v in pred_csv], dtype=float)
        except Exception:
            pass

    # Last-resort for dates: simple index labels to keep UI usable
    if test_dates is None or len(test_dates) == 0:
        test_dates = [str(i) for i in range(len(y_true))]

    # Compute metrics from saved arrays (or prefer persisted values)
    if isinstance(metrics_saved, dict) and metrics_saved:
        metrics = metrics_saved
    else:
        try:
            metrics = compute_metrics_np(y_true.astype(float), y_pred.astype(float))
        except Exception:
            metrics = {}

    model_type = cfg.get("model_type", "transformer")
    model_name = cfg.get("model_name")
    if not model_name:
        model_name = "Transformer" if model_type == "transformer" else "TFT"

    return {
        "metrics": metrics,
        "metrics_steps": cfg.get("metrics_steps"),
        "metrics_train": metrics_train_saved if isinstance(metrics_train_saved, dict) else None,
        "feature_cols": cfg.get("feature_cols"),
        "X_shape": None,
        "y_true": y_true.tolist() if hasattr(y_true, "tolist") else list(y_true or []),
        "y_pred": y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred or []),
        "test_dates": test_dates,
        "future_point": future_point,
        "actual_target": actual_target,
        "future_point_error": future_point_error,
        "f1_by_horizon": f1_by_horizon,
        "rolling_f1": rolling_f1,
        "target": cfg.get("target", "next_close"),
        "residualized": bool(cfg.get("target") == "next_close" and cfg.get("residualize", True)),
        "residual_base": cfg.get("residual_base", "close"),
        "ticker": cfg.get("ticker"),
        "start": cfg.get("start"),
        "end": cfg.get("end"),
        "horizon": cfg.get("horizon"),
        "window_size": cfg.get("window_size") or cfg.get("window"),
        "split_date": cfg.get("split_date"),
        "model_type": model_type,
        "model_name": model_name,
        "run_dir": str(d),
        "split_info": split_info,
        "diagnostics": diagnostics,
    }
