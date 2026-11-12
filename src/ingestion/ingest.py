# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 1 - Data ingestion.

Downloads the MovieLens 100K archive from GroupLens, verifies its MD5
checksum against the value pinned in configs/pipeline.yaml, extracts
the three entity files and rewrites them as canonical UTF-8 CSVs:

    data/raw/users.csv          (user details      - from u.user)
    data/raw/items.csv          (item details      - from u.item)
    data/raw/interactions.csv   (explicit ratings  - from u.data)

An ingestion summary (row counts, byte sizes, checksum, schema) is
written to reports/ingestion_summary.json.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import load_config
from src.utils import get_logger, utc_now_iso, write_json

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

USER_COLS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_BASE_COLS = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
INTERACTION_COLS = ["user_id", "movie_id", "rating", "timestamp"]


def _download(url: str, dest: Path, logger) -> None:
    if dest.exists():
        logger.info("Archive already present: %s (skipping download)", dest.name)
        return
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    logger.info("Downloaded %.1f MB", len(resp.content) / 1e6)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    cfg = load_config()
    raw_dir: Path = cfg["paths"]["data_raw"]
    logger = get_logger("ingestion", cfg["paths"]["logs"])

    # 1. download + checksum verification -------------------------------
    archive = raw_dir / "ml-100k.zip"
    _download(cfg["dataset"]["source_url"], archive, logger)
    digest = _md5(archive)
    if digest != cfg["dataset"]["md5"]:
        raise RuntimeError(
            f"MD5 mismatch for {archive.name}: got {digest}, "
            f"expected {cfg['dataset']['md5']} - aborting ingestion."
        )
    logger.info("MD5 verified: %s", digest)

    # 2. extract ---------------------------------------------------------
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    src_dir = raw_dir / "ml-100k"

    # 3. parse the three entities ---------------------------------------
    users = pd.read_csv(src_dir / "u.user", sep="|", names=USER_COLS,
                        encoding="latin-1", dtype={"zip_code": str})
    items = pd.read_csv(src_dir / "u.item", sep="|",
                        names=ITEM_BASE_COLS + GENRES, encoding="latin-1")
    inter = pd.read_csv(src_dir / "u.data", sep="\t", names=INTERACTION_COLS)

    # 4. write canonical CSVs -------------------------------------------
    out = {
        "users": raw_dir / "users.csv",
        "items": raw_dir / "items.csv",
        "interactions": raw_dir / "interactions.csv",
    }
    users.to_csv(out["users"], index=False)
    items.to_csv(out["items"], index=False)
    inter.to_csv(out["interactions"], index=False)

    summary = {
        "stage": "ingestion",
        "executed_at_utc": utc_now_iso(),
        "source_url": cfg["dataset"]["source_url"],
        "archive_md5": digest,
        "entities": {
            name: {
                "file": str(path.relative_to(cfg["project_root"])),
                "rows": int(len(df)),
                "columns": list(df.columns),
                "bytes": int(path.stat().st_size),
            }
            for name, path, df in [
                ("users", out["users"], users),
                ("items", out["items"], items),
                ("interactions", out["interactions"], inter),
            ]
        },
    }
    write_json(summary, cfg["paths"]["reports"] / "ingestion_summary.json")
    logger.info("Ingestion complete: users=%d items=%d interactions=%d",
                len(users), len(items), len(inter))
    return summary


if __name__ == "__main__":
    run()
