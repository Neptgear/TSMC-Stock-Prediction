"""Data-quality checks for market inputs and saved model runs.

The checks in this module are intentionally conservative: questionable rows are
reported and rejected in strict mode instead of being silently repaired.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


REQUIRED_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def audit_ohlcv(df: pd.DataFrame, *, strict: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate and normalize one OHLCV source without inventing price data.

    Duplicate dates keep the last observation and are recorded. Missing or
    impossible OHLCV rows raise in strict mode; non-strict mode removes them and
    records the removal count.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("OHLCV data is empty or is not a DataFrame.")

    missing_columns = [column for column in REQUIRED_OHLCV if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    clean = df.copy()
    parsed_index = pd.to_datetime(clean.index, errors="coerce")
    invalid_date_rows = int(parsed_index.isna().sum())
    if invalid_date_rows:
        raise ValueError(f"OHLCV index contains {invalid_date_rows} invalid date value(s).")
    clean.index = parsed_index
    clean = clean.sort_index()

    duplicate_dates = int(clean.index.duplicated(keep="last").sum())
    if duplicate_dates:
        clean = clean.loc[~clean.index.duplicated(keep="last")].copy()

    for column in REQUIRED_OHLCV:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean[list(REQUIRED_OHLCV)] = clean[list(REQUIRED_OHLCV)].replace([np.inf, -np.inf], np.nan)

    missing_by_column = {
        column: int(clean[column].isna().sum()) for column in REQUIRED_OHLCV
    }
    missing_rows = int(clean[list(REQUIRED_OHLCV)].isna().any(axis=1).sum())

    positive_price = (clean[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    # Adjusted-price providers can introduce sub-cent floating-point drift, so
    # use a tiny relative tolerance while still rejecting genuine bad bars.
    price_scale = clean[["Open", "High", "Low", "Close"]].abs().max(axis=1).clip(lower=1.0)
    tolerance = price_scale * 1e-6
    high_envelope = clean["High"] + tolerance >= clean[["Open", "Low", "Close"]].max(axis=1)
    low_envelope = clean["Low"] - tolerance <= clean[["Open", "High", "Close"]].min(axis=1)
    nonnegative_volume = clean["Volume"] >= 0
    invalid_market_rows = int((~(positive_price & high_envelope & low_envelope & nonnegative_volume)).sum())

    problems = []
    if missing_rows:
        problems.append(f"{missing_rows} row(s) contain missing OHLCV values")
    if invalid_market_rows:
        problems.append(f"{invalid_market_rows} row(s) violate price/volume constraints")
    if problems and strict:
        raise ValueError("OHLCV audit failed: " + "; ".join(problems))

    removed_rows = 0
    if problems:
        valid_mask = (
            ~clean[list(REQUIRED_OHLCV)].isna().any(axis=1)
            & positive_price
            & high_envelope
            & low_envelope
            & nonnegative_volume
        )
        removed_rows = int((~valid_mask).sum())
        clean = clean.loc[valid_mask].copy()

    if clean.empty:
        raise ValueError("No valid OHLCV rows remain after audit.")

    report: Dict[str, Any] = {
        "status": "passed" if not problems else "passed_after_removal",
        "rows": int(len(clean)),
        "start": clean.index.min().isoformat(),
        "end": clean.index.max().isoformat(),
        "index_monotonic": bool(clean.index.is_monotonic_increasing),
        "index_unique": bool(clean.index.is_unique),
        "duplicate_dates_removed": duplicate_dates,
        "invalid_date_rows": invalid_date_rows,
        "missing_by_column": missing_by_column,
        "invalid_market_rows": invalid_market_rows,
        "rows_removed": removed_rows,
    }
    clean.attrs["data_quality"] = report
    return clean, report


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def audit_saved_run(run_dir: str | Path) -> Dict[str, Any]:
    """Cross-check saved predictions, dates, metrics and split metadata."""
    root = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checks: Dict[str, Any] = {}

    required = ("y_true.npy", "y_pred.npy", "test_dates.json", "metrics.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {
            "passed": False,
            "run_dir": str(root),
            "errors": [f"Missing required files: {missing}"],
            "warnings": warnings,
            "checks": checks,
        }

    y_true = np.asarray(np.load(root / "y_true.npy", allow_pickle=False), dtype=float).reshape(-1)
    y_pred = np.asarray(np.load(root / "y_pred.npy", allow_pickle=False), dtype=float).reshape(-1)
    dates_raw = _load_json(root / "test_dates.json")
    metrics = _load_json(root / "metrics.json")

    dates = pd.to_datetime(pd.Index(dates_raw), errors="coerce")
    lengths = {"y_true": int(y_true.size), "y_pred": int(y_pred.size), "dates": int(len(dates))}
    checks["lengths"] = lengths
    if len(set(lengths.values())) != 1:
        errors.append(f"Prediction/date lengths do not match: {lengths}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        errors.append("Prediction arrays contain NaN or infinite values.")
    if dates.isna().any():
        errors.append("test_dates.json contains invalid dates.")
    else:
        if not dates.is_monotonic_increasing:
            errors.append("Test dates are not in chronological order.")
        if not dates.is_unique:
            errors.append("Test dates contain duplicates.")

    if y_true.size == y_pred.size and y_true.size and np.isfinite(y_true).all() and np.isfinite(y_pred).all():
        diff = y_true - y_pred
        recalculated = {
            "mse": float(np.mean(diff**2)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mae": float(np.mean(np.abs(diff))),
        }
        checks["recalculated_metrics"] = recalculated
        for name, value in recalculated.items():
            stored = metrics.get(name)
            if not isinstance(stored, (int, float)) or not math.isclose(
                float(stored), value, rel_tol=1e-6, abs_tol=1e-8
            ):
                errors.append(f"Stored {name} does not match the saved arrays.")

    config_path = root / "config.json"
    split_path = root / "split_info.json"
    if config_path.is_file() and split_path.is_file():
        config = _load_json(config_path)
        split = _load_json(split_path)
        configured = config.get("train_ratio")
        requested = split.get("requested_train_ratio")
        rule = str(split.get("split_rule", ""))
        ratio_match = re.search(r"ratio_(?:samples|time)_([0-9.]+)", rule)
        recorded = requested if isinstance(requested, (int, float)) else (
            float(ratio_match.group(1)) if ratio_match else None
        )
        checks["configured_train_ratio"] = configured
        checks["split_recorded_train_ratio"] = recorded
        if isinstance(configured, (int, float)) and isinstance(recorded, (int, float)):
            if not math.isclose(float(configured), float(recorded), rel_tol=0, abs_tol=1e-9):
                errors.append(
                    f"config train_ratio={configured} conflicts with split metadata ratio={recorded}."
                )

        if not dates.isna().any() and len(dates):
            expected_start = str(dates.min().date())
            expected_end = str(dates.max().date())
            if split.get("test_start") not in (None, expected_start):
                errors.append("split_info test_start does not match test_dates.json.")
            if split.get("test_end") not in (None, expected_end):
                errors.append("split_info test_end does not match test_dates.json.")
    else:
        warnings.append("config.json or split_info.json is missing; split consistency was not checked.")

    if not (root / "model.pt").is_file():
        warnings.append("model.pt is missing; the run cannot be reproduced for new inference.")
    if not (root / "x_scaler.pkl").is_file():
        warnings.append("x_scaler.pkl is missing; feature transformation cannot be reproduced.")

    return {
        "passed": not errors,
        "run_dir": str(root),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }

