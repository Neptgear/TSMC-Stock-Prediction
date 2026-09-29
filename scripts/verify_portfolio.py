"""Offline verification for the admissions portfolio evidence."""

from __future__ import annotations

import argparse
import compileall
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "app.py",
    "templates/index.html",
    "docs/portfolio-case-study.md",
    "docs/visual-guide.md",
    "docs/results/asof-backtest-2026-09-11.md",
    "docs/results/rolling-models-20x3-2026-09-17.md",
    "docs/assets/05-tsmc-rolling-models.svg",
    "docs/assets/06-tsmc-system-architecture.svg",
)


def verify_files() -> list[str]:
    errors = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {relative}")
    return errors


def verify_evidence() -> list[str]:
    errors = []
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    if "plotly" not in requirements:
        errors.append("requirements.txt does not include Plotly used by app.build_charts")

    report = (ROOT / "docs/results/rolling-models-20x3-2026-09-17.md").read_text(
        encoding="utf-8"
    )
    for phrase in ("120次訓練", "480筆預測", "49.22", "信賴區間"):
        if phrase not in report:
            errors.append(f"rolling-model report is missing evidence phrase: {phrase}")

    for relative in (
        "docs/assets/05-tsmc-rolling-models.svg",
        "docs/assets/06-tsmc-system-architecture.svg",
    ):
        try:
            ET.parse(ROOT / relative)
        except ET.ParseError as exc:
            errors.append(f"invalid SVG {relative}: {exc}")
    return errors


def run_tests() -> bool:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-tests", action="store_true", help="also run the unit-test suite")
    args = parser.parse_args()

    errors = verify_files() + verify_evidence()
    excluded = re.compile(r"[\\/](?:\.venv[^\\/]*|runs|artifacts)[\\/]")
    compiled = compileall.compile_dir(ROOT, quiet=1, rx=excluded)
    if not compiled:
        errors.append("one or more Python files failed to compile")
    if args.with_tests and not run_tests():
        errors.append("unit tests failed")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("PASS: portfolio files, evidence, SVGs, and Python sources are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

