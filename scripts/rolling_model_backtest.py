"""Multi-seed rolling-origin price evaluation for Transformer and TFT-style models."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf

CACHE_DIR = ROOT / ".yf-cache"
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))

import train_transformer as training
from data_fetch import get_ohlcv
from scripts.rolling_price_baselines import evaluate_baselines


def _blocks(data, periods: int, horizon: int):
    first_test_start = len(data) - periods * horizon
    for period in range(periods):
        test_start = first_test_start + period * horizon
        test_end = test_start + horizon
        yield data.iloc[:test_start], data.iloc[test_start:test_end]


def _cached_fetcher(full_data):
    def fetch(ticker_yt="2330.TW", start="2010-01-01", end=None, auto_adjust=True, provider=None):
        if ticker_yt != "2330.TW" or not auto_adjust or provider not in (None, "yahoo"):
            raise ValueError("Rolling model test only permits cached adjusted 2330.TW data")
        frame = full_data.loc[pd_timestamp(start):pd_timestamp(end) if end else None].copy()
        frame.attrs = copy.deepcopy(full_data.attrs)
        return frame

    return fetch


def pd_timestamp(value):
    import pandas as pd

    return pd.Timestamp(value)


def _checkpoint(path: Path, metadata, completed):
    payload = dict(metadata)
    payload["runs"] = completed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize(completed, baseline_report, models, seeds):
    threshold = float(baseline_report["summary"]["naive_last_close"]["mae"])
    rng = np.random.default_rng(20260917)
    summary = {}
    for model in models:
        seed_rows = {}
        for seed in seeds:
            runs = [
                row for row in completed
                if row.get("status") == "ok" and row["model"] == model and row["seed"] == seed
            ]
            errors = np.asarray(
                [error for row in runs for error in row["errors"]], dtype=float
            )
            paired_deltas = np.asarray(
                [
                    model_error - naive_error
                    for row in runs
                    for model_error, naive_error in zip(
                        np.abs(np.asarray(row["errors"], dtype=float)),
                        np.abs(float(row["cutoff_close"]) - np.asarray(row["actual"], dtype=float)),
                    )
                ],
                dtype=float,
            )
            # The four errors inside one forecast period share one cutoff and
            # one fitted model, so they are not independent observations.
            # Resample whole period-level means instead of individual days.
            period_deltas = np.asarray(
                [
                    float(
                        np.mean(
                            np.abs(np.asarray(row["errors"], dtype=float))
                            - np.abs(
                                float(row["cutoff_close"])
                                - np.asarray(row["actual"], dtype=float)
                            )
                        )
                    )
                    for row in runs
                ],
                dtype=float,
            )
            if period_deltas.size:
                bootstrap_means = np.mean(
                    period_deltas[
                        rng.integers(
                            0,
                            period_deltas.size,
                            size=(20000, period_deltas.size),
                        )
                    ],
                    axis=1,
                )
                ci95 = [
                    float(np.percentile(bootstrap_means, 2.5)),
                    float(np.percentile(bootstrap_means, 97.5)),
                ]
            else:
                ci95 = [None, None]
            seed_rows[str(seed)] = {
                "periods": len(runs),
                "observations": int(errors.size),
                "mae": float(np.mean(np.abs(errors))) if errors.size else None,
                "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
                "below_naive_threshold": bool(errors.size and np.mean(np.abs(errors)) < threshold),
                "mae_delta_vs_naive": float(np.mean(paired_deltas)) if paired_deltas.size else None,
                "paired_bootstrap_ci95": ci95,
                "bootstrap_unit": "four_day_period",
                "statistically_clear_improvement": bool(ci95[1] is not None and ci95[1] < 0),
            }
        valid_maes = [row["mae"] for row in seed_rows.values() if row["mae"] is not None]
        all_errors = np.asarray(
            [
                error
                for row in completed
                if row.get("status") == "ok" and row["model"] == model
                for error in row["errors"]
            ],
            dtype=float,
        )
        summary[model] = {
            "seeds": seed_rows,
            "mean_seed_mae": float(np.mean(valid_maes)) if valid_maes else None,
            "std_seed_mae": float(np.std(valid_maes)) if valid_maes else None,
            "pooled_mae": float(np.mean(np.abs(all_errors))) if all_errors.size else None,
            "pooled_rmse": float(np.sqrt(np.mean(all_errors**2))) if all_errors.size else None,
            "stable_below_naive_threshold": bool(
                len(valid_maes) == len(seeds) and all(value < threshold for value in valid_maes)
            ),
            "all_seeds_statistically_clear": bool(
                len(seed_rows) == len(seeds)
                and all(row["statistically_clear_improvement"] for row in seed_rows.values())
            ),
        }
    return threshold, summary


def _markdown(report):
    lines = [
        "# TSMC 20 區間 × 3 種子模型測試",
        "",
        f"- 區間：{report['period_count']} 個互不重疊的 {report['horizon']} 日區間",
        f"- 種子：{', '.join(map(str, report['seeds']))}",
        f"- 通過門檻：每個種子的 MAE 都低於最後收盤基準 {report['naive_mae_threshold']:.2f}",
        f"- 設定：epochs={report['epochs']}、refit={report['refit_epochs']}、d_model={report['d_model']}",
        "",
        "| 模型 | 種子 | MAE | 相對基準 MAE 差 | 95% CI | 低於門檻 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for model, result in report["summary"].items():
        for seed, row in result["seeds"].items():
            passed = "是" if row["below_naive_threshold"] else "否"
            ci = row["paired_bootstrap_ci95"]
            lines.append(
                f"| {model} | {seed} | {row['mae']:.2f} | {row['mae_delta_vs_naive']:+.2f} | "
                f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | {passed} |"
            )
    lines.extend(["", "| 模型 | 種子平均 MAE | 種子標準差 | 穩定通過 |", "| --- | ---: | ---: | --- |"]) 
    for model, result in report["summary"].items():
        passed = "是" if result["stable_below_naive_threshold"] else "否"
        lines.append(
            f"| {model} | {result['mean_seed_mae']:.2f} | {result['std_seed_mae']:.2f} | {passed} |"
        )
    lines.extend(
        [
            "",
            "MAE 差為模型減基準，負值代表模型較好。若 95% CI 跨過 0，表示目前還沒有清楚證據證明差異不是抽樣波動。",
            "",
            "此結果是縮小模型的篩選測試，用來判斷是否值得進行更昂貴的完整訓練；不代表交易績效。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-09-17")
    parser.add_argument("--periods", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--seeds", default="11,42,97")
    parser.add_argument("--models", default="transformer,tft")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--refit-epochs", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--output", default="artifacts/rolling-models-20x3.json")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    output = ROOT / args.output
    full_data = get_ohlcv("2330.TW", args.start, args.end, auto_adjust=True)
    baseline_report = evaluate_baselines(full_data, periods=args.periods, horizon=args.horizon)
    metadata = {
        "method": "expanding-window rolling-origin direct multi-horizon evaluation",
        "ticker": "2330.TW",
        "start": args.start,
        "end": args.end,
        "period_count": int(args.periods),
        "horizon": int(args.horizon),
        "seeds": seeds,
        "models": models,
        "epochs": int(args.epochs),
        "refit_epochs": int(args.refit_epochs),
        "d_model": int(args.d_model),
        "baseline_summary": baseline_report["summary"],
    }
    completed = []
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        comparable = all(existing.get(key) == metadata.get(key) for key in metadata if key != "baseline_summary")
        if comparable:
            completed = existing.get("runs", [])
    for row in completed:
        if row.get("status") == "ok" and "cutoff_close" not in row:
            row["cutoff_close"] = float(full_data.loc[row["cutoff"], "Close"])
    done_keys = {
        (row.get("model"), row.get("seed"), row.get("cutoff"))
        for row in completed if row.get("status") == "ok"
    }

    original_fetcher = training.get_ohlcv
    training.get_ohlcv = _cached_fetcher(full_data)
    total = args.periods * len(seeds) * len(models)
    counter = 0
    try:
        for history, actual in _blocks(full_data, args.periods, args.horizon):
            cutoff = history.index[-1].strftime("%Y-%m-%d")
            actual_dates = tuple(value.strftime("%Y-%m-%d") for value in actual.index)
            actual_values = actual["Close"].to_numpy(dtype=float)
            for seed in seeds:
                for model in models:
                    counter += 1
                    key = (model, seed, cutoff)
                    if key in done_keys:
                        print(f"[{counter}/{total}] skip {model} seed={seed} cutoff={cutoff}", flush=True)
                        continue
                    started = time.perf_counter()
                    print(f"[{counter}/{total}] run {model} seed={seed} cutoff={cutoff}", flush=True)
                    try:
                        cfg = training.TrainConfig(
                            ticker="2330.TW",
                            start=args.start,
                            end=cutoff,
                            target="next_close",
                            horizon=args.horizon,
                            forecast_dates=actual_dates,
                            window_size=120,
                            batch_size=128,
                            epochs=args.epochs,
                            forecast_refit_epochs=args.refit_epochs,
                            lr=5e-4,
                            d_model=args.d_model,
                            nhead=4,
                            num_layers=1,
                            dim_feedforward=args.d_model * 2,
                            dropout=0.1,
                            patience=max(2, args.epochs),
                            loss="huber",
                            selection_metric="mae",
                            random_seed=seed,
                            residualize=True,
                            train_ratio=0.8,
                            val_ratio=0.1,
                            min_test_samples=40,
                            use_fundamentals=False,
                            model_type=model,
                            device="cpu",
                        )
                        result = training.run_training(cfg)
                        predictions = np.asarray(
                            [row["value"] for row in result["future_sequence"]], dtype=float
                        )
                        errors = predictions - actual_values
                        row = {
                            "status": "ok",
                            "model": model,
                            "seed": seed,
                            "cutoff": cutoff,
                            "cutoff_close": float(history["Close"].iloc[-1]),
                            "forecast_dates": list(actual_dates),
                            "predictions": predictions.tolist(),
                            "actual": actual_values.tolist(),
                            "errors": errors.tolist(),
                            "mae": float(np.mean(np.abs(errors))),
                            "rmse": float(np.sqrt(np.mean(errors**2))),
                            "runtime_seconds": float(time.perf_counter() - started),
                        }
                        completed.append(row)
                        done_keys.add(key)
                    except Exception as exc:
                        completed.append(
                            {
                                "status": "error",
                                "model": model,
                                "seed": seed,
                                "cutoff": cutoff,
                                "error": str(exc),
                            }
                        )
                    _checkpoint(output, metadata, completed)
    finally:
        training.get_ohlcv = original_fetcher

    threshold, summary = _summarize(completed, baseline_report, models, seeds)
    report = dict(metadata)
    report.update(
        {
            "naive_mae_threshold": threshold,
            "summary": summary,
            "runs": completed,
        }
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {output}\nMarkdown: {output.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

