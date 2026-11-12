# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Airflow DAG - end-to-end recommendation pipeline orchestration.

Mirrors the lab-session pattern (``airflow standalone`` + BashOperator):
every task calls a stage module of this repository inside the PROJECT
virtualenv, which keeps Airflow's dependency tree isolated from the ML
environment.

Task graph (training of the two models runs in parallel):

  ingest >> validate >> prepare >> transform >> build_features
      >> [train_popularity, train_item_knn] >> evaluate

Required environment variables (exported by scripts/setup_airflow.sh):
  RECSYS_PROJECT_ROOT  - absolute path of this repository
  RECSYS_VENV_PY       - absolute path of .venv/bin/python
"""
from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get("RECSYS_PROJECT_ROOT", os.path.expanduser("~/recommendation-pipeline"))
VENV_PY = os.environ.get("RECSYS_VENV_PY", f"{PROJECT_ROOT}/.venv/bin/python")

ENV = {
    "RECSYS_PROJECT_ROOT": PROJECT_ROOT,
    "PATH": os.environ.get("PATH", ""),
    "HOME": os.environ.get("HOME", ""),
}


def stage(task_id: str, module: str, extra_args: str = "") -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_ROOT} && {VENV_PY} -m {module} {extra_args}",
        env=ENV,
        append_env=True,
    )


with DAG(
    dag_id="recsys_pipeline_2025AG05429",
    description="DM4ML A2 - end-to-end recommendation data pipeline",
    start_date=pendulum.datetime(2026, 11, 1, tz="UTC"),
    schedule="@weekly",      # periodic retraining; also triggered manually
    catchup=False,
    default_args={
        "owner": "2025AG05429",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["dm4ml", "recommendation", "2025AG05429"],
) as dag:
    ingest = stage("ingest", "src.ingestion.ingest")
    validate = stage("validate", "src.validation.validate")
    prepare = stage("prepare", "src.preparation.prepare")
    transform = stage("transform", "src.transformation.transform")
    features = stage("build_features", "src.features.build_features")
    train_pop = stage("train_popularity", "src.training.train", "--model popularity")
    train_knn = stage("train_item_knn", "src.training.train", "--model item_knn")
    evaluate = stage("evaluate", "src.evaluation.evaluate")

    ingest >> validate >> prepare >> transform >> features
    features >> [train_pop, train_knn] >> evaluate
