#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if curl --fail --silent http://127.0.0.1:5050/health >/dev/null; then
  echo "Demo already running on 5050."; exit 0
fi
nohup .venv-cloud/bin/gunicorn --bind 0.0.0.0:5050 --workers 1 --threads 2 --timeout 180 --error-logfile cloud-demo.log cloud_demo:app >/dev/null 2>&1 </dev/null &
echo "Starting real TSMC demo: http://localhost:5050/"
