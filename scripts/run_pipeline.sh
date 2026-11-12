#!/usr/bin/env bash
# Roll No: 2025AG05429 | DM4ML Assignment 2
# Runs the full pipeline end-to-end (no Airflow needed), then the tests.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python -m src.pipeline
echo
echo "[run] pipeline finished - running unit tests"
.venv/bin/python -m pytest tests/ -q
echo
echo "[run] key outputs:"
echo "  reports/validation_report.md"
echo "  reports/feature_catalog.md"
echo "  reports/evaluation/model_comparison.md"
echo "  mlruns/            (open with: .venv/bin/mlflow ui --backend-store-uri file:./mlruns)"
