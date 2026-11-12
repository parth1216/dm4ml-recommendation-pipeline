# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Shared helpers: logging and small JSON utilities."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_dir: Path) -> logging.Logger:
    """Logger that writes to the console AND logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured (module re-import)
        return logger
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(stream)

    log_dir.mkdir(parents=True, exist_ok=True)
    fileh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    fileh.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(fileh)
    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
