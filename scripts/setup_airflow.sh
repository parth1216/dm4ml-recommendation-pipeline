#!/usr/bin/env bash
# Roll No: 2025AG05429 | DM4ML Assignment 2
# Installs Apache Airflow 2.10.5 in its OWN virtualenv (.venv-airflow),
# wires this repo's dags/ folder in, and starts `airflow standalone`
# exactly like the DM4ML lab session.
#
# Usage:
#   bash scripts/setup_airflow.sh            # install + start UI at :8080
#   bash scripts/setup_airflow.sh install    # install only
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

AIRFLOW_VERSION=2.10.5
PY=${PYTHON_BIN:-python3}
PYVER=$($PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
CONSTRAINTS="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYVER}.txt"

if [ ! -d .venv-airflow ]; then
  $PY -m venv .venv-airflow
fi
.venv-airflow/bin/pip install --upgrade pip >/dev/null

# Fetch the official constraints file; fall back to the codeload tarball
# for networks where raw.githubusercontent.com does not resolve.
CONSTRAINTS_FILE=".venv-airflow/constraints-${AIRFLOW_VERSION}-py${PYVER}.txt"
if [ ! -s "${CONSTRAINTS_FILE}" ]; then
  echo "[airflow] fetching constraints (py${PYVER}) ..."
  if ! curl -fsSL --connect-timeout 15 "${CONSTRAINTS}" -o "${CONSTRAINTS_FILE}"; then
    echo "[airflow] raw.githubusercontent.com unreachable - using codeload fallback"
    TARBALL="https://codeload.github.com/apache/airflow/tar.gz/refs/tags/constraints-${AIRFLOW_VERSION}"
    curl -fsSL --retry 3 --retry-delay 2 "${TARBALL}" -o "${CONSTRAINTS_FILE}.tar.gz"
    tar -xzOf "${CONSTRAINTS_FILE}.tar.gz" \
      "airflow-constraints-${AIRFLOW_VERSION}/constraints-${PYVER}.txt" > "${CONSTRAINTS_FILE}"
    rm -f "${CONSTRAINTS_FILE}.tar.gz"
  fi
fi
.venv-airflow/bin/pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINTS_FILE}"

# ---- Airflow home lives inside the repo (nothing leaks to ~) ---------
export AIRFLOW_HOME="${PROJECT_ROOT}/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export RECSYS_PROJECT_ROOT="${PROJECT_ROOT}"
export RECSYS_VENV_PY="${PROJECT_ROOT}/.venv/bin/python"

echo "[airflow] AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "[airflow] DAGs folder = ${AIRFLOW__CORE__DAGS_FOLDER}"

if [ "${1:-start}" = "install" ]; then
  echo "[airflow] install-only mode done."
  exit 0
fi

echo "[airflow] starting standalone (UI: http://localhost:8080)"
echo "[airflow] admin password will be in ${AIRFLOW_HOME}/standalone_admin_password.txt"
exec .venv-airflow/bin/airflow standalone
