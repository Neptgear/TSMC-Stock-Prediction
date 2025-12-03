import os
import json
import datetime as dt
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

from data_fetch import get_ohlcv, get_fundamentals_timeseries, get_event_countdowns
from ta_features import compute_features, compute_indicators_only, get_default_feature_columns
from train_transformer import TrainConfig, run_training, standardize_ohlcv_columns, suggest_window
from runs_utils import list_runs as list_saved_runs, load_run_results


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "devkey")

    def _hyperparams_for_h(h: int) -> Dict[str, Any]:
        """Preset hyperparameters tuned by horizon buckets."""
        h_eff = int(max(1, h))
        if h_eff <= 3:
            return dict(
                epochs=12,
                batch_size=64,
                lr=2e-4,
                d_model=48,
                nhead=8,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.50,
                patience=6,
                weight_decay=3.5e-3,
                loss="huber",
                huber_delta=0.5,
                clip_grad_norm=0.25,
                train_ratio=0.6,
                split_mode="ratio_time",
                test_days=0,
                walkforward_splits=8,
                walkforward_epochs=12,
            )
        if h_eff <= 5:
            return dict(
                epochs=20,
                batch_size=64,
                lr=2e-4,
                d_model=48,
                nhead=8,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.50,
                patience=6,
                weight_decay=5e-3,
                loss="huber",
                huber_delta=0.7,
                clip_grad_norm=0.25,
                train_ratio=0.8,
                split_mode="ratio_time",
                test_days=0,
                walkforward_splits=4,
                walkforward_epochs=10,
            )
        if h_eff <= 7:
            return dict(
                epochs=25,
                batch_size=64,
                lr=1e-4,
                d_model=48,
                nhead=8,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.50,
                patience=8,
                weight_decay=7.5e-3,
                loss="huber",
                huber_delta=1.0,
                clip_grad_norm=0.25,
                train_ratio=0.8,
                split_mode="ratio_time",
                test_days=0,
                walkforward_splits=4,
                walkforward_epochs=10,
            )
        return dict(
            epochs=25,
            batch_size=64,
            lr=1e-4,
            d_model=48,
            nhead=8,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.55,
            patience=8,
            weight_decay=1e-2,
            loss="huber",
            huber_delta=1.0,
            clip_grad_norm=0.25,
            train_ratio=0.82,
            split_mode="ratio_time",
            test_days=0,
            walkforward_splits=4,
            walkforward_epochs=12,
        )

    def _recommend_start_for_h(h: int) -> str:
        """Suggest a historical start date based on horizon length."""
        years_back = 7 if h <= 3 else 9 if h <= 10 else 12
        today = dt.date.today()
        start_date = today - dt.timedelta(days=365 * years_back)
        return start_date.isoformat()

    @app.route("/", methods=["GET", "POST"])
    def index():
        # Read params from GET/POST
        req = request.values
        ticker = req.get("ticker", "2330.TW")
        # Default start date shown in the UI
        default_start = "2022-01-01"
        start = req.get("start", default_start)
        end = req.get("end", None)
        # Always predict next_close (remove next_return_pct option)
        target = "next_close"
        horizon = int(req.get("horizon", 1))
        # Residualization controls for price target
        residualize = req.get("residualize", "1") == "1"
        residual_base = req.get("residual_base", "close")
        window = int(req.get("window", 30))
        epochs = int(req.get("epochs", 30))
        batch_size = int(req.get("batch_size", 64))
        lr = float(req.get("lr", 1e-3))
        d_model = int(req.get("d_model", 128))
        nhead = int(req.get("nhead", 4))
        num_layers = int(req.get("num_layers", 3))
        dim_feedforward = int(req.get("dim_feedforward", 256))
        dropout = float(req.get("dropout", 0.1))
        patience = int(req.get("patience", 10))
        weight_decay = float(req.get("weight_decay", 1e-5))
        loss = req.get("loss", "huber")
        huber_delta = float(req.get("huber_delta", 0.5))
        clip_grad_norm = float(req.get("clip_grad_norm", 0.0))
        train_ratio = float(req.get("train_ratio", 0.6))
        split_mode = req.get("split_mode", "ratio_time")
        test_days = int(req.get("test_days", 0))
        walkforward_splits = int(req.get("walkforward_splits", 2))
        walkforward_epochs = req.get("walkforward_epochs", "")
        walkforward_epochs = int(walkforward_epochs) if str(walkforward_epochs).strip() != "" else None
        # Default scale_target to Yes; we also force scaling in code for h>1
        scale_target = req.get("scale_target", "1") == "1"
        uf_raw = req.get("use_fundamentals", "")
        # Default to Yes; only explicit falsy tokens disable fundamentals
        use_fundamentals = str(uf_raw).lower() not in ("0", "false", "no", "off")
        min_samples_fund = int(req.get("min_samples_fund", 30))
        preset = req.get("preset", "custom")
        # Optional explicit split date for train/test cutoff
        split_date = req.get("split_date", "")
        run_id = req.get("run_id", "")
        action = req.get("do", "view")

        # Apply hyperparameter presets when using a predefined horizon/window preset
        if preset != "custom":
            hp = _hyperparams_for_h(horizon)
            epochs = int(hp["epochs"])
            batch_size = int(hp["batch_size"])
            lr = float(hp["lr"])
            d_model = int(hp["d_model"])
            nhead = int(hp["nhead"])
            num_layers = int(hp["num_layers"])
            dim_feedforward = int(hp["dim_feedforward"])
            dropout = float(hp["dropout"])
            patience = int(hp["patience"])
            weight_decay = float(hp.get("weight_decay", weight_decay))
            loss = hp["loss"]
            huber_delta = float(hp["huber_delta"])
            clip_grad_norm = float(hp["clip_grad_norm"])
            train_ratio = float(hp["train_ratio"])
            split_mode = hp["split_mode"]
            test_days = int(hp["test_days"])
            walkforward_splits = int(hp["walkforward_splits"])
            walkforward_epochs = int(hp["walkforward_epochs"])

        recommended_start = _recommend_start_for_h(horizon)

        # Fetch OHLCV and precompute indicators for charts (robust to horizon)
        df = get_ohlcv(ticker_yt=ticker, start=start, end=end, auto_adjust=True)
        df = standardize_ohlcv_columns(df)
        events_df = get_event_countdowns(ticker, df.index)
        feat_for_charts = compute_indicators_only(df, include_extra=True)

        # Plan feature lists for this dataset (so UI can show even before training)
        # Start from the chart features (pure technicals) so we always have something.
        base_cols = get_default_feature_columns(feat_for_charts)
        planned_feature_cols_tech: list[str] = [c for c in base_cols if not (isinstance(c, str) and c.startswith("f_"))]
        planned_feature_cols_fund: list[str] = []
        # Best-effort refinement using the full training feature pipeline (with fundamentals)
        try:
            tmp_feat = compute_features(
                df, target=target, horizon=horizon, include_extra=True, events_df=events_df
            )
            if use_fundamentals:
                try:
                    fund_df = get_fundamentals_timeseries(ticker, align_index=df.index)
                    if isinstance(fund_df, pd.DataFrame) and not fund_df.empty:
                        overlap = [c for c in fund_df.columns if c in tmp_feat.columns]
                        if overlap:
                            fund_df = fund_df.drop(columns=overlap)
                        tmp_feat = tmp_feat.join(fund_df, how="left")
                        fcols = [c for c in tmp_feat.columns if isinstance(c, str) and c.startswith("f_")]
                        good_f = [c for c in fcols if tmp_feat[c].notna().sum() >= 4]
                        drop_f = [c for c in fcols if c not in good_f]
                        if drop_f:
                            tmp_feat = tmp_feat.drop(columns=drop_f)
                except Exception:
                    pass
            tmp_cols = get_default_feature_columns(tmp_feat)
            # Replace tech list with the training-time tech order
            planned_feature_cols_tech = [c for c in tmp_cols if not (isinstance(c, str) and c.startswith("f_"))]
            planned_feature_cols_fund = [c for c in tmp_cols if isinstance(c, str) and c.startswith("f_")]
        except Exception:
            # If this fails, we keep the base technical list and no fundamentals.
            planned_feature_cols_fund = planned_feature_cols_fund

        # Prepare charts (Close with MAs, Volume, RSI, MACD)
        charts = build_charts(feat_for_charts)

        train_result: Optional[Dict[str, Any]] = None
        alert: Optional[str] = None
        multi_results: Optional[list] = None
        runs_list = list_saved_runs()
        if action == "train_unused_legacy":
            try:
                cfg = TrainConfig(
                    ticker=ticker,
                    start=start,
                    end=end,
                    target=target,
                    horizon=horizon,
                    window_size=window,
                    epochs=epochs,
                    scale_target=scale_target,
                    residualize=residualize,
                    residual_base=residual_base,
                    split_date=(split_date or None),
                    use_fundamentals=bool(use_fundamentals),
                    require_fundamentals=False,
                    min_samples_with_fundamentals=int(min_samples_fund),
                )
                train_result = run_training(cfg)
                # Convert numpy arrays to lists for JSON embedding (if present)
                if isinstance(train_result.get("y_true"), np.ndarray):
                    train_result["y_true"] = train_result["y_true"].tolist()
                if isinstance(train_result.get("y_pred"), np.ndarray):
                    train_result["y_pred"] = train_result["y_pred"].tolist()
            except Exception as ex:  # surface friendly alert when history is insufficient
                # Compute a helpful message with required history estimate
                try:
                    required_days = 252 + max(0, window - 1) + max(1, horizon)
                    if len(df.index) <= required_days:
                        first_target = df.index[0] + pd.tseries.offsets.BDay(required_days)
                        alert = (
                            f"Not enough history for window={window}, horizon={horizon}. "
                            f"Need ≈{required_days} trading days after start. "
                            f"Earliest target date would be {getattr(first_target, 'date', lambda: first_target)()} — "
                            f"pick an earlier Start, smaller Window, or lower Horizon."
                        )
                    else:
                        alert = f"Training failed: {type(ex).__name__}: {ex}"
                except Exception:
                    alert = f"Training failed: {type(ex).__name__}: {ex}"

        # Compare removed

        def _build_cfg(model_type: str) -> TrainConfig:
            # Common training config for both architectures.
            # Use the exact window chosen in the UI (or preset)
            # for both Transformer and TFT so the comparison is fair.
            eff_window = window

            cfg_kwargs: Dict[str, Any] = dict(
                ticker=ticker,
                start=start,
                end=end,
                target=target,
                horizon=horizon,
                window_size=eff_window,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                patience=patience,
                weight_decay=weight_decay,
                loss=loss,
                huber_delta=huber_delta,
                clip_grad_norm=clip_grad_norm,
                train_ratio=train_ratio,
                split_mode=split_mode,
                test_days=test_days,
                walkforward_splits=walkforward_splits,
                walkforward_epochs=walkforward_epochs if walkforward_epochs is not None else max(3, min(10, epochs // 2)) if epochs else 3,
                scale_target=scale_target,
                residualize=residualize,
                residual_base=residual_base,
                split_date=(split_date or None),
                use_fundamentals=bool(use_fundamentals),
                require_fundamentals=False,
                min_samples_with_fundamentals=int(min_samples_fund),
                model_type=model_type,
            )
            # For TFT we bias the optimiser and regularisation towards
            # multi-step quantile forecasting and directly minimising
            # absolute price differences while keeping training stable.
            if model_type.lower() == "tft":
                # Use quantile loss (0.1/0.5/0.9); scale targets and
                # keep residualisation so absolute price errors remain small.
                cfg_kwargs["loss"] = "mse"  # ignored for TFT; QuantileLoss is used instead
                cfg_kwargs["quantiles"] = (0.1, 0.5, 0.9)
                # Always enable target scaling for numerical stability.
                cfg_kwargs["scale_target"] = True
                # Encoder/hidden configuration tuned per horizon. We tie this
                # to `horizon` (and thus indirectly to the preset) so that
                # short, medium, and long horizons get appropriate capacity.
                h_local = int(max(1, horizon))
                d_model_local = cfg_kwargs.get("d_model", d_model)
                num_layers_local = cfg_kwargs.get("num_layers", num_layers)
                weight_decay_local = 1e-4
                patience_val = max(8, epochs // 3)
                if h_local <= 3:
                    # Short-term: smaller model, stronger regularisation.
                    d_model_local = 64
                    num_layers_local = 2
                    weight_decay_local = 1e-4
                    patience_val = max(8, epochs // 3)
                elif h_local <= 10:
                    # Medium-term (e.g., h=5–10): more capacity and slightly
                    # looser regularisation.
                    d_model_local = 96
                    num_layers_local = 2
                    weight_decay_local = 5e-5
                    patience_val = max(12, epochs // 2)
                else:
                    # Longer horizons: largest capacity and longer patience.
                    d_model_local = 128
                    num_layers_local = 3
                    weight_decay_local = 5e-5
                    patience_val = max(15, int(max(10, 0.6 * epochs)))

                cfg_kwargs["d_model"] = d_model_local
                cfg_kwargs["nhead"] = 4
                cfg_kwargs["num_layers"] = num_layers_local
                cfg_kwargs["dropout"] = 0.2
                # Training dynamics: LR shared, weight decay / patience vary.
                cfg_kwargs["lr"] = 1e-3
                cfg_kwargs["weight_decay"] = weight_decay_local
                cfg_kwargs["patience"] = patience_val
                cfg_kwargs["clip_grad_norm"] = 0.1
                # Override horizon-specific tuning with the original
                # fixed TFT configuration for consistency.
                cfg_kwargs["d_model"] = 48
                cfg_kwargs["nhead"] = 4
                cfg_kwargs["num_layers"] = 2
                cfg_kwargs["dropout"] = 0.2
                cfg_kwargs["lr"] = 1e-3
                cfg_kwargs["weight_decay"] = 1e-4
                cfg_kwargs["patience"] = 8
                cfg_kwargs["clip_grad_norm"] = 0.1
            return TrainConfig(**cfg_kwargs)

        def _format_train_error(ex: Exception) -> str:
            try:
                required_days = 252 + max(0, window - 1) + max(1, horizon)
                if len(df.index) <= required_days:
                    first_target = df.index[0] + pd.tseries.offsets.BDay(required_days)
                    return (
                        f"Not enough history for window={window}, horizon={horizon}. "
                        f"Need at least {required_days} trading days after start. "
                        f"Earliest target date would be {getattr(first_target, 'date', lambda: first_target)()}; "
                        f"pick an earlier Start, smaller Window, or lower Horizon."
                    )
                return f"Training failed: {type(ex).__name__}: {ex}"
            except Exception:
                return f"Training failed: {type(ex).__name__}: {ex}"

        def _log_metrics(prefix: str, tr: Optional[Dict[str, Any]]) -> None:
            try:
                if not tr:
                    return
                metrics = tr.get("metrics") or {}
                model_name = tr.get("model_name") or tr.get("model_type") or "model"
                print(f"[metrics] {prefix} {model_name}:")
                for k, v in metrics.items():
                    try:
                        print(f"[metrics]   {k}: {float(v):.6f}")
                    except Exception:
                        print(f"[metrics]   {k}: {v}")
            except Exception:
                pass

        if action in ("train", "train_transformer"):
            try:
                cfg = _build_cfg("transformer")
                train_result = run_training(cfg)
                if isinstance(train_result.get("y_true"), np.ndarray):
                    train_result["y_true"] = train_result["y_true"].tolist()
                if isinstance(train_result.get("y_pred"), np.ndarray):
                    train_result["y_pred"] = train_result["y_pred"].tolist()
                train_result["model_type"] = "transformer"
                train_result.setdefault("model_name", "Transformer")
                # Attach basic context for clearer labels in the UI
                train_result.setdefault("ticker", ticker)
                train_result.setdefault("start", start)
                train_result.setdefault("end", end)
                train_result.setdefault("split_date", split_date or "")
                train_result.setdefault("horizon", horizon)
                train_result.setdefault("window_size", window)
                _log_metrics("train", train_result)
            except Exception as ex:
                alert = _format_train_error(ex)

        if action == "train_tft":
            try:
                cfg = _build_cfg("tft")
                train_result = run_training(cfg)
                if isinstance(train_result.get("y_true"), np.ndarray):
                    train_result["y_true"] = train_result["y_true"].tolist()
                if isinstance(train_result.get("y_pred"), np.ndarray):
                    train_result["y_pred"] = train_result["y_pred"].tolist()
                train_result["model_type"] = "tft"
                train_result.setdefault("model_name", "TFT")
                train_result.setdefault("ticker", ticker)
                train_result.setdefault("start", start)
                train_result.setdefault("end", end)
                train_result.setdefault("split_date", split_date or "")
                train_result.setdefault("horizon", horizon)
                train_result.setdefault("window_size", window)
                _log_metrics("train", train_result)
            except Exception as ex:
                alert = _format_train_error(ex)

        if action == "train_both":
            multi_results = []
            for m_type, m_name in (("transformer", "Transformer"), ("tft", "TFT")):
                try:
                    cfg = _build_cfg(m_type)
                    res = run_training(cfg)
                    if isinstance(res.get("y_true"), np.ndarray):
                        res["y_true"] = res["y_true"].tolist()
                    if isinstance(res.get("y_pred"), np.ndarray):
                        res["y_pred"] = res["y_pred"].tolist()
                    res["model_type"] = m_type
                    res.setdefault("model_name", m_name)
                    res.setdefault("ticker", ticker)
                    res.setdefault("start", start)
                    res.setdefault("end", end)
                    res.setdefault("split_date", split_date or "")
                    res.setdefault("horizon", horizon)
                    res.setdefault("window_size", window)
                    _log_metrics("train_both", res)
                    multi_results.append(res)
                except Exception as ex:
                    alert = _format_train_error(ex)
                    multi_results = None
                    break
            if multi_results:
                train_result = multi_results[0]

        # Load saved run results (no re-predict)
        if action in ("predict", "load") and run_id:
            try:
                pr = load_run_results(run_dir=f"runs/{run_id}")
                model_type_loaded = pr.get("model_type", "transformer")
                model_name_loaded = pr.get("model_name")
                if not model_name_loaded:
                    model_name_loaded = "Transformer" if model_type_loaded == "transformer" else "TFT"
                train_result = {
                    "metrics": pr.get("metrics", {}),
                    "metrics_train": pr.get("metrics_train"),
                    "y_true": pr.get("y_true", []),
                    "y_pred": pr.get("y_pred", []),
                    "test_dates": pr.get("test_dates", []),
                    "target": pr.get("target", target),
                    "future_point": pr.get("future_point"),
                    "actual_target": pr.get("actual_target"),
                    "future_point_error": pr.get("future_point_error"),
                    "feature_cols": pr.get("feature_cols", []),
                    "run_dir": f"runs/{run_id}",
                    "ticker": pr.get("ticker", ticker),
                    "start": pr.get("start", start),
                    "end": pr.get("end", end),
                    "split_date": pr.get("split_date", split_date or ""),
                    "horizon": pr.get("horizon", horizon),
                    "window_size": pr.get("window_size", window),
                    "model_type": model_type_loaded,
                    "model_name": model_name_loaded,
                    "split_info": pr.get("split_info"),
                    "f1_by_horizon": pr.get("f1_by_horizon"),
                    "rolling_f1": pr.get("rolling_f1"),
                    "diagnostics": pr.get("diagnostics"),
                }
            except Exception as ex:
                alert = f"Predict failed: {type(ex).__name__}: {ex}"

        # Derive indicator lists used for UI when we have a result
        if isinstance(train_result, dict):
            try:
                fc_tech = train_result.get("feature_cols_tech")
                fc_fund = train_result.get("feature_cols_fund")
                if fc_tech is None or fc_fund is None:
                    fc = train_result.get("feature_cols") or []
                    fc_tech = [c for c in fc if isinstance(c, str) and not c.startswith("f_")]
                    fc_fund = [c for c in fc if isinstance(c, str) and c.startswith("f_")]
                    train_result["feature_cols_tech"] = fc_tech
                    train_result["feature_cols_fund"] = fc_fund
            except Exception:
                pass

        # Save current run artifacts to disk (on demand)
        if action == "save_run":
            try:
                tr_json = req.get("tr_json")
                if not tr_json:
                    raise RuntimeError("No run payload provided.")
                payload = json.loads(tr_json)

                def _save_single_run(tr_dict: Dict[str, Any], idx: int = 0) -> str:
                    import time as _time, numpy as _np
                    suffix = tr_dict.get("model_type", f"m{idx}")
                    run_dir_local = os.path.join("runs", f"saved_run_{_time.strftime('%Y%m%d_%H%M%S')}_{suffix}")
                    os.makedirs(run_dir_local, exist_ok=True)
                    # Arrays and dates
                    _np.save(os.path.join(run_dir_local, "y_true.npy"), _np.array(tr_dict.get("y_true", []), dtype=float))
                    _np.save(os.path.join(run_dir_local, "y_pred.npy"), _np.array(tr_dict.get("y_pred", []), dtype=float))
                    with open(os.path.join(run_dir_local, "test_dates.json"), "w", encoding="utf-8") as f:
                        json.dump(tr_dict.get("test_dates", []), f)
                    # Config and metrics
                    cfg_save_local = {
                        "ticker": tr_dict.get("ticker", ticker),
                        "start": tr_dict.get("start", start),
                        "end": tr_dict.get("end", end),
                        "target": tr_dict.get("target", target),
                        "horizon": int(tr_dict.get("horizon", horizon)),
                        "window_size": int(tr_dict.get("window_size", window)),
                        "epochs": epochs,
                        "scale_target": bool(tr_dict.get("scale_target", scale_target)),
                        "residualize": bool(tr_dict.get("residualize", residualize)),
                        "residual_base": tr_dict.get("residual_base", residual_base),
                        "split_date": tr_dict.get("split_date", split_date or None),
                        "model_type": tr_dict.get("model_type", "transformer"),
                        "batch_size": tr_dict.get("batch_size", batch_size),
                        "lr": tr_dict.get("lr", lr),
                        "d_model": tr_dict.get("d_model", d_model),
                        "nhead": tr_dict.get("nhead", nhead),
                        "num_layers": tr_dict.get("num_layers", num_layers),
                        "dim_feedforward": tr_dict.get("dim_feedforward", dim_feedforward),
                        "dropout": tr_dict.get("dropout", dropout),
                        "patience": tr_dict.get("patience", patience),
                        "loss": tr_dict.get("loss", loss),
                        "huber_delta": tr_dict.get("huber_delta", huber_delta),
                        "clip_grad_norm": tr_dict.get("clip_grad_norm", clip_grad_norm),
                        "weight_decay": tr_dict.get("weight_decay", weight_decay),
                        "train_ratio": tr_dict.get("train_ratio", train_ratio),
                        "split_mode": tr_dict.get("split_mode", split_mode),
                        "test_days": tr_dict.get("test_days", test_days),
                        "walkforward_splits": tr_dict.get("walkforward_splits", walkforward_splits),
                        "walkforward_epochs": tr_dict.get("walkforward_epochs", walkforward_epochs),
                        "feature_cols": tr_dict.get("feature_cols"),
                    }
                    with open(os.path.join(run_dir_local, "config.json"), "w", encoding="utf-8") as f:
                        json.dump(cfg_save_local, f, indent=2)
                    with open(os.path.join(run_dir_local, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(tr_dict.get("metrics", {}), f, indent=2)
                    try:
                        mtrain = tr_dict.get("metrics_train")
                        if isinstance(mtrain, dict) and mtrain:
                            with open(os.path.join(run_dir_local, "metrics_train.json"), "w", encoding="utf-8") as f:
                                json.dump(mtrain, f, indent=2)
                    except Exception:
                        pass
                    try:
                        sinfo = tr_dict.get("split_info")
                        if isinstance(sinfo, dict) and sinfo:
                            with open(os.path.join(run_dir_local, "split_info.json"), "w", encoding="utf-8") as f:
                                json.dump(sinfo, f, indent=2)
                        diagnostics = tr_dict.get("diagnostics")
                        if isinstance(diagnostics, dict) and diagnostics:
                            with open(os.path.join(run_dir_local, "diagnostics.json"), "w", encoding="utf-8") as f:
                                json.dump(diagnostics, f, indent=2)
                    except Exception:
                        pass
                    # Optional future prediction point and actual target (for re-loading in UI)
                    try:
                        future_point_local = tr_dict.get("future_point")
                        if isinstance(future_point_local, dict):
                            with open(os.path.join(run_dir_local, "future_point.json"), "w", encoding="utf-8") as f:
                                json.dump(future_point_local, f, indent=2)
                        future_point_err_local = tr_dict.get("future_point_error")
                        if future_point_err_local:
                            with open(os.path.join(run_dir_local, "future_point_error.json"), "w", encoding="utf-8") as f:
                                json.dump({"error": future_point_err_local}, f, indent=2)
                    except Exception:
                        pass
                    try:
                        actual_target_local = tr_dict.get("actual_target")
                        if isinstance(actual_target_local, dict):
                            with open(os.path.join(run_dir_local, "actual_target.json"), "w", encoding="utf-8") as f:
                                json.dump(actual_target_local, f, indent=2)
                    except Exception:
                        pass
                    try:
                        f1_h = tr_dict.get("f1_by_horizon")
                        if f1_h:
                            with open(os.path.join(run_dir_local, "f1_by_horizon.json"), "w", encoding="utf-8") as f:
                                json.dump(f1_h, f, indent=2)
                    except Exception:
                        pass
                    try:
                        roll_f1 = tr_dict.get("rolling_f1")
                        if roll_f1:
                            with open(os.path.join(run_dir_local, "rolling_f1.json"), "w", encoding="utf-8") as f:
                                json.dump(roll_f1, f, indent=2)
                    except Exception:
                        pass
                    return run_dir_local

                saved_dirs = []
                if isinstance(payload, list):
                    for idx, item in enumerate(payload):
                        if isinstance(item, dict):
                            saved_dirs.append(_save_single_run(item, idx))
                    # Use the first as the active train_result
                    if saved_dirs and isinstance(payload[0], dict):
                        payload[0]["run_dir"] = saved_dirs[0]
                        train_result = payload[0]
                        run_id = os.path.basename(saved_dirs[0])
                    alert = f"Saved {len(saved_dirs)} runs."
                else:
                    run_dir = _save_single_run(payload)
                    payload["run_dir"] = run_dir
                    train_result = payload
                    run_id = os.path.basename(run_dir)
                    alert = f"Run saved: {run_id}"
            except Exception as ex:
                alert = f"Save run failed: {type(ex).__name__}: {ex}"

        # Delete selected run directory
        if action == "delete_run" and run_id:
            try:
                path = os.path.join("runs", run_id)
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                    alert = f"Deleted run {run_id}."
                else:
                    alert = f"Run {run_id} not found."
            except Exception as ex:
                alert = f"Delete failed: {type(ex).__name__}: {ex}"

        # Delete all runs
        if action == "delete_all_runs":
            try:
                import shutil
                root = "runs"
                if os.path.isdir(root):
                    for name in os.listdir(root):
                        p = os.path.join(root, name)
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                    alert = "Deleted all runs."
            except Exception as ex:
                alert = f"Delete all failed: {type(ex).__name__}: {ex}"

        # Refresh runs list after any save/delete
        runs_list = list_saved_runs()

        # Save predictions to DB (per-run table)
        if action == "save_db":
            try:
                tr_json = req.get("tr_json")
                if not tr_json:
                    raise RuntimeError("No prediction payload provided.")
                tr = json.loads(tr_json)
                dates = tr.get("test_dates", [])
                y_true = tr.get("y_true", [])
                y_pred = tr.get("y_pred", [])
                if not (len(dates) == len(y_true) == len(y_pred)):
                    raise RuntimeError("Mismatched lengths in prediction arrays.")
                # Determine run identifier to segregate data per run
                run_dir_from_tr = tr.get("run_dir") if isinstance(tr, dict) else None
                if not run_id and run_dir_from_tr:
                    try:
                        run_id = str(run_dir_from_tr).rstrip("/\\").split("/")[-1].split("\\")[-1]
                    except Exception:
                        pass
                if not run_id:
                    run_id = f"adhoc_{dt.datetime.now(dt.UTC).strftime('%Y%m%d_%H%M%S')}"

                saved_at = dt.datetime.now(dt.UTC).isoformat()

                # Save to SQLite with one table per run
                import sqlite3, re

                def _sanitize(name: str) -> str:
                    t = re.sub(r"[^0-9a-zA-Z_]", "_", name)
                    if not t:
                        t = "run"
                    if t[0].isdigit():
                        t = "_" + t
                    return t[:64]

                table = _sanitize(run_id)
                db_path = os.path.join("runs", "predictions.db")
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS [{table}] (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          date TEXT NOT NULL,
                          true REAL,
                          pred REAL,
                          ticker TEXT,
                          target TEXT,
                          horizon INTEGER,
                          window INTEGER,
                          residualize INTEGER,
                          residual_base TEXT,
                          scale_target INTEGER,
                          run_id TEXT,
                          start TEXT,
                          end TEXT,
                          saved_at TEXT
                        )
                        """
                    )
                    conn.execute(
                        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_date ON [{table}](date)"
                    )
                    rows = [
                        (
                            str(dates[i]), float(y_true[i]) if y_true[i] is not None else None,
                            float(y_pred[i]) if y_pred[i] is not None else None,
                            str(ticker), str(target), int(horizon), int(window), int(bool(residualize)),
                            str(residual_base), int(bool(scale_target)), str(run_id), str(start), str(end or ""), saved_at,
                        )
                        for i in range(len(dates))
                    ]
                    conn.executemany(
                        f"INSERT OR REPLACE INTO [{table}] (date,true,pred,ticker,target,horizon,window,residualize,residual_base,scale_target,run_id,start,end,saved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        rows,
                    )
                    conn.commit()
                finally:
                    conn.close()
                alert = f"Saved {len(dates)} rows to predictions.db table [{table}]."
                # Keep showing the same result on the page
                train_result = {
                    "metrics": tr.get("metrics", {}),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "test_dates": dates,
                    "target": tr.get("target", target),
                    "future_point": tr.get("future_point"),
                    "actual_target": tr.get("actual_target"),
                }
            except Exception as ex:
                alert = f"Save failed: {type(ex).__name__}: {ex}"

        # Delete runs removed

        # Clear CSV removed

        # Metric hover help (shown as browser tooltips)
        metrics_help = {
            "mse": "MSE (Mean Squared Error)\nWhat: Average squared error.\nGood/Bad: Lower is better; scale depends on target units.",
            "rmse": "RMSE (Root Mean Squared Error)\nWhat: Typical error in target units.\nGood/Bad: Lower is better; as a % of price <1% excellent, 1-2% good, 2-5% fair, >5% weak.",
            "mae": "MAE (Mean Absolute Error)\nWhat: Average absolute error in target units.\nGood/Bad: Lower is better; as a % of price <1% excellent, 1-2% good, 2-5% fair, >5% weak.",
            "mape": "MAPE (Mean Absolute Percentage Error)\nWhat: Average percent error.\nGood/Bad: <1% excellent, 1-2% good, 2-5% fair, >5% weak. Caution near zero true values.",
            "r2": "R2 (Coefficient of Determination)\nWhat: Variance explained vs. mean predictor.\nGood/Bad: >0.80 excellent, 0.60-0.80 good, 0.30-0.60 fair, 0-0.30 weak, <0 poor.",
            "overfit_gap_rmse": "Overfit Gap (RMSE)\nWhat: Test RMSE minus train RMSE.\nGood/Bad: Near 0 is ideal; large positive suggests overfitting, negative can signal underfit or data drift.",
            "direction_acc": "Directional Accuracy\nWhat: Correct up/down calls fraction.\nGood/Bad: >60% good, 55-60% fair, ~50% weak (coinflip).",
            "precision_up": "Precision (Up)\nWhat: Of predicted ups, how many were truly up.\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "recall_up": "Recall (Up)\nWhat: Of true ups, how many we predicted up.\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "f1_up": "F1 (Up)\nWhat: Balance of up precision and recall.\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "precision_down": "Precision (Down)\nWhat: Of predicted downs, how many were truly down.\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "recall_down": "Recall (Down)\nWhat: Of true downs, how many we predicted down.\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "f1_down": "F1 (Down)\nWhat: Balance of down precision and recall.\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "macro_f1": "Macro-F1\nWhat: Average of F1_up and F1_down (class-balanced).\nGood/Bad: >60% good, 55-60% fair, <55% weak.",
            "weighted_f1": "Weighted-F1\nWhat: F1 weighted by class frequency.\nGood/Bad: Track alongside Macro-F1 to spot majority-class bias.",
            "f1_stability": "F1 Stability\nWhat: |train F1 - test F1| (macro if available).\nGood/Bad: <0.03 excellent, <0.05 good, <0.10 fair, higher suggests overfit.",
        }

        # Per-metric quick analysis shown under values and in tooltips
        def _grade_pct(pct: float) -> str:
            if pct < 1.0:
                return "Excellent"
            if pct < 2.0:
                return "Good"
            if pct < 5.0:
                return "Fair"
            if pct < 10.0:
                return "Weak"
            return "Poor"

        def _grade_prob(v: float) -> str:
            if v >= 0.6:
                return "Good"
            if v >= 0.55:
                return "Fair"
            return "Weak"

        metrics_analysis: Dict[str, str] = {}
        if isinstance(train_result, dict) and train_result and isinstance(train_result.get("metrics"), dict):
            m = train_result["metrics"]
            tgt = train_result.get("target", target)
            try:
                import numpy as _np
                yt = _np.array(train_result.get("y_true", []), dtype=float)
                med = float(_np.nanmedian(yt)) if yt.size else float("nan")
            except Exception:
                med = float("nan")

            # RMSE/MAE as percent for next_close; for returns treat as percentage directly
            rmse = float(m.get("rmse", float("nan")))
            mae = float(m.get("mae", float("nan")))
            if tgt == "next_close" and med and med == med and med != 0:
                rmse_pct = 100.0 * rmse / med
                mae_pct = 100.0 * mae / med
                metrics_analysis["rmse"] = f"{_grade_pct(rmse_pct)}: ~{rmse_pct:.2f}% of price"
                metrics_analysis["mae"] = f"{_grade_pct(mae_pct)}: ~{mae_pct:.2f}% of price"
            else:
                # next_return_pct (fraction); display in %
                rmse_pct = 100.0 * rmse
                mae_pct = 100.0 * mae
                metrics_analysis["rmse"] = f"{_grade_pct(rmse_pct)}: ~{rmse_pct:.2f}% returns"
                metrics_analysis["mae"] = f"{_grade_pct(mae_pct)}: ~{mae_pct:.2f}% returns"

            # MAPE
            mape = m.get("mape")
            if isinstance(mape, (int, float)):
                metrics_analysis["mape"] = f"{_grade_pct(float(mape))}: {float(mape):.2f}%"

            # R2
            r2 = m.get("r2")
            if isinstance(r2, (int, float)):
                rv = float(r2)
                if rv > 0.8:
                    g = "Excellent"
                elif rv > 0.6:
                    g = "Good"
                elif rv > 0.3:
                    g = "Fair"
                elif rv >= 0.0:
                    g = "Weak"
                else:
                    g = "Poor"
                metrics_analysis["r2"] = f"{g}: R2={rv:.3f}"
            gap_rmse = m.get("overfit_gap_rmse")
            if isinstance(gap_rmse, (int, float)):
                if gap_rmse > 0:
                    note = "Test error higher; possible overfit"
                elif gap_rmse < 0:
                    note = "Test error lower; watch for underfit/data shift"
                else:
                    note = "Train/test match"
                metrics_analysis["overfit_gap_rmse"] = f"{note}: {gap_rmse:.4f}"


            # Directional metrics
            for key in (
                "direction_acc",
                "precision_up",
                "recall_up",
                "f1_up",
                "precision_down",
                "recall_down",
                "f1_down",
                "macro_f1",
                "weighted_f1",
            ):
                v = m.get(key)
                if isinstance(v, (int, float)):
                    metrics_analysis[key] = f"{_grade_prob(float(v))}: {float(v)*100:.1f}%"
            fs = m.get("f1_stability")
            if isinstance(fs, (int, float)):
                gap = float(fs)
                if gap < 0.03:
                    band = "Excellent"
                elif gap < 0.05:
                    band = "Good"
                elif gap < 0.1:
                    band = "Fair"
                else:
                    band = "Weak"
                metrics_analysis["f1_stability"] = f"{band}: gap={gap:.3f}"

        # Always render the page (both view and train flows)
            return render_template(
                "index.html",
            ticker=ticker,
            start=start,
            end=end,
            target=target,
            horizon=horizon,
            window=window,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            patience=patience,
            loss=loss,
            huber_delta=huber_delta,
            clip_grad_norm=clip_grad_norm,
            train_ratio=train_ratio,
            split_mode=split_mode,
            test_days=test_days,
            walkforward_splits=walkforward_splits,
            walkforward_epochs=walkforward_epochs,
            scale_target=scale_target,
            residualize=residualize,
            residual_base=residual_base,
            preset=preset,
            split_date=split_date,
            recommended_start=recommended_start,
            run_id=run_id,
            use_fundamentals=use_fundamentals,
            min_samples_fund=min_samples_fund,
            runs_list=runs_list,
            charts_json=json.dumps(charts),
            planned_feature_cols_tech=planned_feature_cols_tech,
            planned_feature_cols_fund=planned_feature_cols_fund,
            train_result=train_result,
            multi_results=multi_results,
            alert=alert,
            metrics_help=metrics_help,
            metrics_analysis=metrics_analysis,
        )

        # Render page for both view and train flows
        return render_template(
            "index.html",
            ticker=ticker,
            start=start,
            end=end,
            target=target,
            horizon=horizon,
            window=window,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            patience=patience,
            loss=loss,
            huber_delta=huber_delta,
            clip_grad_norm=clip_grad_norm,
            train_ratio=train_ratio,
            split_mode=split_mode,
            test_days=test_days,
            walkforward_splits=walkforward_splits,
            walkforward_epochs=walkforward_epochs,
            scale_target=scale_target,
            residualize=residualize,
            residual_base=residual_base,
            preset=preset,
            split_date=split_date,
            charts_json=json.dumps(charts),
            recommended_start=recommended_start,
            run_id=run_id,
            use_fundamentals=use_fundamentals,
            min_samples_fund=min_samples_fund,
            runs_list=runs_list,
            planned_feature_cols_tech=planned_feature_cols_tech,
            planned_feature_cols_fund=planned_feature_cols_fund,
            train_result=train_result,
            multi_results=multi_results,
            alert=alert,
            metrics_help=metrics_help,
            metrics_analysis=metrics_analysis,
        )

    return app


def build_charts(feat_df: pd.DataFrame) -> Dict[str, Any]:
    """Prepare chart data consumable by Plotly in the frontend."""
    import plotly.graph_objs as go
    from plotly.utils import PlotlyJSONEncoder

    charts: Dict[str, Any] = {}
    idx = feat_df.index

    # Price + MAs
    price_traces = [
        go.Scatter(x=idx, y=feat_df["Close"], name="Close", mode="lines")
    ]
    for ma_col, name in [("ma5", "MA5"), ("ma10", "MA10"), ("ma20", "MA20")]:
        if ma_col in feat_df.columns:
            price_traces.append(go.Scatter(x=idx, y=feat_df[ma_col], name=name, mode="lines"))
    price_layout = go.Layout(title="Price & Moving Averages", legend={"orientation": "h"})
    price_fig = go.Figure(data=price_traces, layout=price_layout)
    charts["price"] = json.loads(price_fig.to_json())

    # Volume
    if "Volume" in feat_df.columns:
        vol_fig = go.Figure(data=[go.Bar(x=idx, y=feat_df["Volume"], name="Volume")], layout=go.Layout(title="Volume"))
        charts["volume"] = json.loads(vol_fig.to_json())

    # RSI
    if "rsi14" in feat_df.columns:
        rsi_fig = go.Figure(
            data=[go.Scatter(x=idx, y=feat_df["rsi14"], name="RSI14", mode="lines")],
            layout=go.Layout(title="RSI(14)", yaxis={"range": [0, 100]}),
        )
        charts["rsi"] = json.loads(rsi_fig.to_json())

    # MACD
    if set(["macd", "macd_signal", "macd_hist"]).issubset(set(feat_df.columns)):
        macd_fig = go.Figure(
            data=[
                go.Scatter(x=idx, y=feat_df["macd"], name="MACD", mode="lines"),
                go.Scatter(x=idx, y=feat_df["macd_signal"], name="Signal", mode="lines"),
                go.Bar(x=idx, y=feat_df["macd_hist"], name="Hist"),
            ],
            layout=go.Layout(title="MACD"),
        )
        charts["macd"] = json.loads(macd_fig.to_json())

    return charts


if __name__ == "__main__":
    app = create_app()
    # Run with: python app.py (then open http://127.0.0.1:5000)
    app.run(debug=True)
