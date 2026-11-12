#!/usr/bin/env bash
# Roll No: 2025AG05429 | DM4ML Assignment 2
# One-shot environment setup for the ML pipeline (fresh Ubuntu 22.04+).
# Usage:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON_BIN:-python3}
echo "[setup] using interpreter: $($PY --version)"

if [ ! -d .venv ]; then
  $PY -m venv .venv
  echo "[setup] created .venv"
fi
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import pandas, pandera, sklearn, mlflow, scipy; print('[setup] all imports OK')"
echo "[setup] done. Run the pipeline with:  .venv/bin/python -m src.pipeline"
