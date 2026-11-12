# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 3a - Data preparation.

Cleans the validated entities, standardises formats and produces the
chronological train/test split:

  * users   : gender upper-cased, occupation lower-cased, age_group bin
  * items   : release_date parsed to ISO, release_year recovered from the
              title "(YYYY)" suffix when the date is missing, pipe-joined
              ``genres`` string built from the 19 one-hot flags
  * ratings : unix timestamp -> UTC datetime; a user re-rating the same
              movie keeps only the MOST RECENT rating (policy in config)
  * split   : time-based - interactions are sorted by timestamp and the
              first 80% become train, the last 20% test (no leakage).

Outputs: data/processed/{users,items,train,test}.parquet and
reports/preparation_summary.json.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.ingestion.ingest import GENRES
from src.utils import get_logger, utc_now_iso, write_json


def _clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.copy()
    users["gender"] = users["gender"].str.strip().str.upper()
    users["occupation"] = users["occupation"].str.strip().str.lower()
    users["age_group"] = pd.cut(
        users["age"], bins=[0, 17, 24, 34, 44, 54, 64, 200],
        labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype(str)
    return users


def _clean_items(items: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    items["release_date"] = pd.to_datetime(
        items["release_date"], format="%d-%b-%Y", errors="coerce")
    year_from_title = items["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Float64")
    items["release_year"] = (
        items["release_date"].dt.year.astype("Float64").fillna(year_from_title).astype("Int64")
    )
    items["genres"] = items[GENRES].apply(
        lambda row: "|".join(g for g in GENRES if row[g] == 1) or "unknown", axis=1)
    return items.drop(columns=["video_release_date", "imdb_url"])


def run() -> dict:
    cfg = load_config()
    logger = get_logger("preparation", cfg["paths"]["logs"])
    proc = cfg["paths"]["data_processed"]

    users = _clean_users(pd.read_parquet(proc / "validated_users.parquet"))
    items = _clean_items(pd.read_parquet(proc / "validated_items.parquet"))
    inter = pd.read_parquet(proc / "validated_interactions.parquet")

    # ---- standardise interactions ------------------------------------
    inter["event_time"] = pd.to_datetime(inter["timestamp"], unit="s", utc=True)
    n_before = len(inter)
    if cfg["preparation"]["duplicate_policy"] == "keep_latest":
        inter = (inter.sort_values("timestamp")
                      .drop_duplicates(subset=["user_id", "movie_id"], keep="last"))
    n_dupes = n_before - len(inter)

    # ---- chronological train/test split ------------------------------
    test_frac = cfg["preparation"]["test_fraction"]
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    split_ts = inter["timestamp"].quantile(1 - test_frac, interpolation="lower")
    train = inter[inter["timestamp"] <= split_ts]
    test = inter[inter["timestamp"] > split_ts]

    users.to_parquet(proc / "users.parquet", index=False)
    items.to_parquet(proc / "items.parquet", index=False)
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)

    test_users = set(test["user_id"])
    known_users = test_users & set(train["user_id"])
    summary = {
        "stage": "preparation",
        "executed_at_utc": utc_now_iso(),
        "duplicate_user_item_pairs_resolved": int(n_dupes),
        "duplicate_policy": cfg["preparation"]["duplicate_policy"],
        "split": {
            "method": "time-based (global timestamp quantile)",
            "split_timestamp_utc": str(pd.to_datetime(split_ts, unit="s", utc=True)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_window": [str(train["event_time"].min()), str(train["event_time"].max())],
            "test_window": [str(test["event_time"].min()), str(test["event_time"].max())],
            "test_users": len(test_users),
            "test_users_seen_in_train": len(known_users),
            "cold_start_test_users": len(test_users - known_users),
        },
    }
    write_json(summary, cfg["paths"]["reports"] / "preparation_summary.json")
    logger.info("Preparation complete: train=%d test=%d (split at %s), %d re-ratings resolved",
                len(train), len(test), summary["split"]["split_timestamp_utc"], n_dupes)
    return summary


if __name__ == "__main__":
    run()
