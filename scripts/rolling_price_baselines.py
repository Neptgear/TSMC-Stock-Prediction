"""Rolling-origin evaluation for price baselines using adjusted closes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yfinance as yf

CACHE_DIR = ROOT / ".yf-cache"
CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(str(CACHE_DIR))

from data_fetch import get_ohlcv
from scripts.asof_backtest import _price_baselines


def evaluate_baselines(data, periods: int = 20, horizon: int = 4):
    required = periods * horizon + 60
    if len(data) < required:
        raise ValueError(f"Need at least {required} rows, received {len(data)}")

    first_test_start = len(data) - periods * horizon
    aggregate = {}
    period_rows = []

    for period in range(periods):
        test_start = first_test_start + period * horizon
        test_end = test_start + horizon
        history = data.iloc[:test_start]
        actual = data.iloc[test_start:test_end]
        dates = [{"date": value.strftime("%Y-%m-%d")} for value in actual.index]
        baselines = _price_baselines(history, dates, float(history["Close"].iloc[-1]))
        actual_values = actual["Close"].to_numpy(dtype=float)
        period_metrics = {}

        for name, baseline in baselines.items():
            predictions = np.asarray(
                [row["value"] for row in baseline["forecast"]], dtype=float
            )
            errors = predictions - actual_values
            metrics = {
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
            }
            period_metrics[name] = metrics
            slot = aggregate.setdefault(name, {"errors": [], "period_mae": []})
            slot["errors"].extend(errors.tolist())
            slot["period_mae"].append(metrics["mae"])

        winner = min(period_metrics, key=lambda name: period_metrics[name]["mae"])
        period_rows.append(
            {
                "cutoff": history.index[-1].strftime("%Y-%m-%d"),
                "test_start": actual.index[0].strftime("%Y-%m-%d"),
                "test_end": actual.index[-1].strftime("%Y-%m-%d"),
                "winner": winner,
                "metrics": period_metrics,
            }
        )

    summary = {}
    for name, values in aggregate.items():
        errors = np.asarray(values["errors"], dtype=float)
        summary[name] = {
            "observations": int(errors.size),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_period_mae": float(np.mean(values["period_mae"])),
            "median_period_mae": float(np.median(values["period_mae"])),
            "period_wins": int(sum(row["winner"] == name for row in period_rows)),
        }

    ranking = sorted(summary, key=lambda name: summary[name]["mae"])
    return {"summary": summary, "ranking_by_mae": ranking, "periods": period_rows}


def _markdown(report):
    lines = [
        "# TSMC 價格基準 rolling-origin 測試",
        "",
        f"- 結束日期：{report['end']}",
        f"- 區間：{report['period_count']} 個互不重疊的 {report['horizon']} 日區間",
        "- 每個區間的預測只使用該截止日以前的調整後收盤價。",
        "",
        "| 排名 | 方法 | MAE | RMSE | 區間勝出次數 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, name in enumerate(report["ranking_by_mae"], start=1):
        item = report["summary"][name]
        lines.append(
            f"| {rank} | {name} | {item['mae']:.2f} | {item['rmse']:.2f} | "
            f"{item['period_wins']}/{report['period_count']} |"
        )
    lines.extend(
        [
            "",
            "此報告用來設定深度模型必須超越的價格基準，不代表任何交易策略可獲利。",
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
    parser.add_argument("--output", default="artifacts/rolling-price-baselines.json")
    args = parser.parse_args()

    data = get_ohlcv("2330.TW", args.start, args.end, auto_adjust=True)
    report = evaluate_baselines(data, periods=args.periods, horizon=args.horizon)
    report.update(
        {
            "ticker": "2330.TW",
            "end": args.end,
            "period_count": int(args.periods),
            "horizon": int(args.horizon),
            "price_type": "Yahoo Finance adjusted close",
        }
    )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "ranking": report["ranking_by_mae"]}, ensure_ascii=False, indent=2))
    print(f"\nJSON: {output}\nMarkdown: {output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

