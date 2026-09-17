"""Command-line audit for a saved experiment directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_quality import audit_saved_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit saved predictions, metrics and split metadata.")
    parser.add_argument("run_dir", help="Path to one directory under runs/")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    report = audit_saved_run(args.run_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

