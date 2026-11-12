# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Sequential end-to-end runner (orchestration fallback).

Runs every stage in dependency order with timing and fail-fast
behaviour. The production orchestration is the Airflow DAG in
dags/recsys_pipeline_dag.py, which calls the SAME stage entry points;
this runner lets a reviewer execute the whole pipeline with one command:

    python -m src.pipeline
"""
from __future__ import annotations

import time

from src.config import load_config
from src.utils import get_logger


def main() -> None:
    cfg = load_config()
    logger = get_logger("pipeline", cfg["paths"]["logs"])

    from src.ingestion import ingest
    from src.validation import validate
    from src.preparation import prepare
    from src.transformation import transform
    from src.features import build_features
    from src.training import train
    from src.evaluation import evaluate

    stages = [
        ("1-ingestion", ingest.run),
        ("2-validation", validate.run),
        ("3-preparation", prepare.run),
        ("4-transformation", transform.run),
        ("5-features", build_features.run),
        ("6-training", lambda: train.run("all")),
        ("7-evaluation", evaluate.run),
    ]
    t_start = time.time()
    for name, fn in stages:
        t0 = time.time()
        logger.info("=== stage %s started ===", name)
        fn()
        logger.info("=== stage %s finished in %.1fs ===", name, time.time() - t0)
    logger.info("PIPELINE COMPLETE in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
