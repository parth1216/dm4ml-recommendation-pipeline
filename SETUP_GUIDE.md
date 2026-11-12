# Full Step-by-Step Setup & Execution Guide (fresh Ubuntu)

**Roll No: 2025AG05429 | DM4ML Assignment 2**

Verified on Ubuntu 22.04 LTS; works the same on 20.04/24.04. Total time:
~10 minutes on a normal connection (most of it is the one-time Airflow install).

## 0. Prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl
python3 --version   # needs >= 3.10 (3.12 recommended; both venvs use this interpreter)
```

> On Windows, run everything inside WSL2 Ubuntu — identical commands.

## 1. Get the code

```bash
git clone https://github.com/parth1216/dm4ml-recommendation-pipeline.git recommendation-pipeline
cd recommendation-pipeline
```

## 2. Create the ML environment (pinned)

```bash
bash scripts/setup.sh
```

Creates `.venv/`, installs `requirements.txt` (pandas, pandera, scikit-learn,
scipy, mlflow, matplotlib, reportlab, pytest, ...) and verifies imports.

## 3. Run the pipeline end-to-end + tests

```bash
bash scripts/run_pipeline.sh
```

What happens, in order (all stages log to `logs/pipeline.log`):

1. **Ingestion** — downloads `ml-100k.zip` (≈5 MB) from GroupLens, verifies the
   pinned MD5, writes `data/raw/{users,items,interactions}.csv`.
2. **Validation** — 31 Pandera/custom checks; quarantined rows (if any) in
   `data/errors/`, report in `reports/validation_report.md`.
3. **Preparation** — cleaning + chronological 80/20 split →
   `data/processed/{train,test}.parquet`.
4. **Transformation** — user activity, item popularity, sparse user×item matrix.
5. **Features** — Parquet feature store in `data/features/`, catalog in
   `reports/feature_catalog.md`.
6. **Training** — popularity baseline + item-kNN CF; MLflow runs under `mlruns/`.
7. **Evaluation** — metrics table in `reports/evaluation/model_comparison.md`.
8. Unit tests: `8 passed`.

Expected final log line: `PIPELINE COMPLETE in ~2s` (after the first download).

## 4. Inspect MLflow tracking

```bash
.venv/bin/mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Open <http://localhost:5000> → experiment **dm4ml_a2_recsys** → two runs
(`popularity_baseline`, `item_knn_cf`) with params, training stats and the
evaluation metrics appended to the same runs.

## 5. Orchestrated run with Apache Airflow (lab-session style)

```bash
bash scripts/setup_airflow.sh      # installs Airflow 2.10.5 into .venv-airflow and starts standalone
```

* UI: <http://localhost:8080> — username `admin`, password printed in the
  terminal and stored at `.airflow/standalone_admin_password.txt`.
* In the UI: un-pause **recsys_pipeline_2025AG05429** → trigger ▶ → watch the
  Graph view go green: `ingest → validate → prepare → transform →
  build_features → [train_popularity ∥ train_item_knn] → evaluate`.

Headless alternative (no browser):

```bash
export AIRFLOW_HOME=$PWD/.airflow AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags \
       RECSYS_PROJECT_ROOT=$PWD RECSYS_VENV_PY=$PWD/.venv/bin/python
.venv-airflow/bin/airflow dags test recsys_pipeline_2025AG05429
```

## 6. Clean slate (prove reproducibility)

```bash
rm -rf data/raw/* data/processed/* data/features/* data/errors/* mlruns models/* logs/*
bash scripts/run_pipeline.sh     # everything rebuilds identically
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python3 -m venv` fails | `sudo apt install -y python3-venv` |
| Airflow constraints download fails (`raw.githubusercontent.com` blocked) | the script automatically falls back to `codeload.github.com`; both URLs blocked → download `constraints-2.10.5/constraints-<py>.txt` on another machine into `.venv-airflow/` |
| Port 8080 busy | `.venv-airflow/bin/airflow standalone` honours `AIRFLOW__WEBSERVER__WEB_SERVER_PORT=8081` |
| Port 5000 busy (mlflow) | add `--port 5001` |
| Corporate proxy | export `https_proxy` before `scripts/setup.sh` |
