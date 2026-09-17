"""Leakage-safe, as-of multi-day forecast evaluation.

The model receives market data only through --cutoff. Actual prices after the
cutoff are fetched only after all forecasts have been produced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf

CACHE_DIR = ROOT / ".yf-cache"
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))

from data_fetch import get_ohlcv
from train_transformer import TrainConfig, run_training


def _direction(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _evaluate_sequence(sequence, actual_adjusted, actual_raw, cutoff_close):
    rows = []
    previous_prediction = float(cutoff_close)
    previous_actual = float(cutoff_close)
    for item in sequence or []:
        date = item["date"]
        if date not in actual_adjusted.index:
            rows.append({"date": date, "prediction": float(item["value"]), "status": "actual_unavailable"})
            previous_prediction = float(item["value"])
            continue
        adjusted_value = float(actual_adjusted.loc[date, "Close"])
        raw_value = float(actual_raw.loc[date, "Close"]) if date in actual_raw.index else None
        prediction = float(item["value"])
        pred_change = prediction - previous_prediction
        actual_change = adjusted_value - previous_actual
        now_taipei = dt.datetime.now(ZoneInfo("Asia/Taipei"))
        row_date = dt.date.fromisoformat(date)
        is_final = row_date < now_taipei.date() or (
            row_date == now_taipei.date() and now_taipei.time() >= dt.time(13, 40)
        )
        rows.append(
            {
                "date": date,
                "prediction_adjusted_close": prediction,
                "actual_adjusted_close": adjusted_value,
                "actual_exchange_close": raw_value,
                "error": prediction - adjusted_value,
                "absolute_error": abs(prediction - adjusted_value),
                "predicted_daily_direction": _direction(pred_change),
                "actual_daily_direction": _direction(actual_change),
                "daily_direction_correct": _direction(pred_change) == _direction(actual_change),
                "actual_status": "final" if is_final else "provisional",
            }
        )
        previous_prediction = prediction
        previous_actual = adjusted_value

    comparable = [
        row for row in rows
        if "actual_adjusted_close" in row and row.get("actual_status") == "final"
    ]
    errors = np.asarray([row["error"] for row in comparable], dtype=float)
    return {
        "rows": rows,
        "metrics": {
            "days_compared": len(comparable),
            "mae": float(np.mean(np.abs(errors))) if errors.size else None,
            "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "direction_accuracy": (
                float(np.mean([row["daily_direction_correct"] for row in comparable]))
                if comparable
                else None
            ),
        },
    }


def _price_baselines(training_data, forecast_dates, cutoff_close):
    """Forecasts that a learned model must beat on MAE/RMSE."""
    closes = training_data["Close"].astype(float)

    def constant(value):
        return [{"date": item["date"], "value": float(value)} for item in forecast_dates]

    linear_window = min(60, len(closes))
    y = closes.iloc[-linear_window:].to_numpy(dtype=float)
    x = np.arange(linear_window, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    linear_values = intercept + slope * np.arange(
        linear_window, linear_window + len(forecast_dates), dtype=float
    )

    return {
        "naive_last_close": {
            "forecast": constant(cutoff_close),
            "description": "每一天都預測為截止日調整後收盤價",
        },
        "sma_5": {
            "forecast": constant(closes.tail(5).mean()),
            "description": "每一天都預測為截止日前五日平均收盤價",
        },
        "sma_20": {
            "forecast": constant(closes.tail(20).mean()),
            "description": "每一天都預測為截止日前二十日平均收盤價",
        },
        "linear_trend_60": {
            "forecast": [
                {"date": item["date"], "value": float(value)}
                for item, value in zip(forecast_dates, linear_values)
            ],
            "description": "以截止日前六十日收盤價線性趨勢外推",
        },
    }


def _markdown(report):
    lines = [
        "# TSMC as-of 四日預測回測",
        "",
        f"- 資料截止：{report['cutoff']}",
        f"- 預測期間：{report['forecast_start']} 至 {report['forecast_end']}",
        "- 主要目標：最低樣本外調整後收盤價 MAE；RMSE 為次要指標。",
        "- 模型以驗證集 MAE 選擇 checkpoint，Transformer 使用 Huber loss 訓練。",
        "- 模型輸入使用 Yahoo Finance 調整後 OHLCV；交易所原始收盤價另列供參考。",
        "- 實際值只在模型完成預測後讀取，不參與訓練、縮放或特徵計算。",
        "",
    ]
    for model_name, model in report["models"].items():
        lines.extend(
            [
                f"## {model_name}",
                "",
                "| 日期 | 預測調整收盤 | 實際調整收盤 | 交易所收盤 | 絕對誤差 | 日方向 |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in model["evaluation"]["rows"]:
            if "actual_adjusted_close" not in row:
                lines.append(f"| {row['date']} | {row['prediction']:.2f} | 尚無 | 尚無 | 尚無 | 尚無 |")
                continue
            direction = "正確" if row["daily_direction_correct"] else "錯誤"
            if row.get("actual_status") != "final":
                direction = f"{direction}（盤中暫定）"
            raw = "—" if row["actual_exchange_close"] is None else f"{row['actual_exchange_close']:.2f}"
            lines.append(
                f"| {row['date']} | {row['prediction_adjusted_close']:.2f} | "
                f"{row['actual_adjusted_close']:.2f} | {raw} | {row['absolute_error']:.2f} | {direction} |"
            )
        metrics = model["evaluation"]["metrics"]
        lines.extend(
            [
                "",
                f"MAE：{metrics['mae']:.2f}；RMSE：{metrics['rmse']:.2f}；"
                f"日方向正確率：{metrics['direction_accuracy']:.1%}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解讀限制",
            "",
            "這是單一四日封存回測，不足以證明模型可獲利。正式評估仍需多個互不重疊期間、基準模型、交易成本及重複種子。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2026-09-11")
    parser.add_argument("--actual-end", default="2026-09-17")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="transformer,tft")
    parser.add_argument("--output", default="artifacts/asof-backtest-2026-09-11.json")
    args = parser.parse_args()

    model_names = [value.strip() for value in args.models.split(",") if value.strip()]
    training_data = get_ohlcv("2330.TW", args.start, args.cutoff, auto_adjust=True)
    cutoff_close = float(training_data["Close"].iloc[-1])

    # Forecast first. Do not request any post-cutoff observations before this loop ends.
    model_results = {}
    for model_name in model_names:
        config = TrainConfig(
            ticker="2330.TW",
            start=args.start,
            end=args.cutoff,
            target="next_close",
            horizon=args.horizon,
            window_size=120,
            batch_size=64,
            epochs=args.epochs,
            lr=5e-4,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            patience=4,
            loss="huber",
            selection_metric="mae",
            random_seed=args.seed,
            residualize=True,
            train_ratio=0.8,
            val_ratio=0.1,
            min_test_samples=60,
            use_fundamentals=False,
            model_type=model_name,
            device="cpu",
            forecast_refit_epochs=6,
        )
        result = run_training(config)
        if result.get("future_point_error") or not result.get("future_sequence"):
            raise RuntimeError(f"{model_name} forecast failed: {result.get('future_point_error')}")
        model_results[model_name] = {
            "forecast": result["future_sequence"],
            "historical_test_metrics": result.get("metrics"),
            "split_info": result.get("split_info"),
        }

    # Only now load the hidden period for scoring.
    actual_adjusted = get_ohlcv("2330.TW", args.cutoff, args.actual_end, auto_adjust=True)
    actual_raw = get_ohlcv("2330.TW", args.cutoff, args.actual_end, auto_adjust=False)
    actual_adjusted.index = actual_adjusted.index.strftime("%Y-%m-%d")
    actual_raw.index = actual_raw.index.strftime("%Y-%m-%d")
    for value in model_results.values():
        value["evaluation"] = _evaluate_sequence(
            value["forecast"], actual_adjusted, actual_raw, cutoff_close
        )

    forecast_dates = model_results[model_names[0]]["forecast"]
    for baseline_name, baseline in _price_baselines(
        training_data, forecast_dates, cutoff_close
    ).items():
        baseline["evaluation"] = _evaluate_sequence(
            baseline["forecast"], actual_adjusted, actual_raw, cutoff_close
        )
        model_results[baseline_name] = baseline
    report = {
        "method": "sealed as-of direct multi-horizon backtest",
        "primary_objective": "lowest out-of-sample adjusted-close MAE, RMSE as secondary",
        "model_selection_metric": "validation_mae",
        "optimization_loss": "huber",
        "random_seed": int(args.seed),
        "ticker": "2330.TW",
        "cutoff": args.cutoff,
        "forecast_start": forecast_dates[0]["date"],
        "forecast_end": forecast_dates[-1]["date"],
        "cutoff_adjusted_close": cutoff_close,
        "training_rows": int(len(training_data)),
        "training_data_quality": training_data.attrs.get("data_quality"),
        "models": model_results,
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nJSON: {output}\nMarkdown: {markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

