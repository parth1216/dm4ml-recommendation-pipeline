# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Central configuration loader.

Every pipeline stage calls :func:`load_config` so that paths and
hyper-parameters live in exactly one file (``configs/pipeline.yaml``).
The project root can be overridden with the ``RECSYS_PROJECT_ROOT``
environment variable, which is how the Airflow DAG points the tasks at
the checked-out repository.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(
    os.environ.get("RECSYS_PROJECT_ROOT", Path(__file__).resolve().parents[1])
)

_PATH_KEYS = (
    "data_raw",
    "data_processed",
    "data_features",
    "data_errors",
    "models",
    "reports",
    "logs",
    "mlruns",
)


def load_config() -> dict:
    """Load configs/pipeline.yaml and resolve all paths to absolute Paths."""
    cfg_file = PROJECT_ROOT / "configs" / "pipeline.yaml"
    with open(cfg_file, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["project_root"] = PROJECT_ROOT
    for key in _PATH_KEYS:
        p = PROJECT_ROOT / cfg["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = p
    return cfg
