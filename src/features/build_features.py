# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 4 - Feature management.

Computes user, item and interaction features FROM THE TRAIN WINDOW ONLY
(reference date = last train timestamp, so the features are exactly what
would have been available at prediction time) and writes them to the
Parquet-backed feature store under data/features/.

Each feature is declared once in the FEATURE_CATALOG registry below; the
catalog table shipped with the repo (reports/feature_catalog.md|csv) is
GENERATED from that registry, so code and documentation cannot drift.
"""
from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.ingestion.ingest import GENRES
from src.utils import get_logger, utc_now_iso, write_json

FEATURE_CATALOG = [
    # entity, name, dtype, description, computation
    ("user", "user_n_ratings", "int", "Activity count: number of train ratings by the user", "count over train window"),
    ("user", "user_avg_rating", "float", "Mean rating the user gives", "mean(rating) over train window"),
    ("user", "user_rating_std", "float", "Spread of the user's ratings (0 if single rating)", "std(rating), filled with 0"),
    ("user", "user_days_since_last", "float", "Recency: days between user's last train rating and the reference date", "(ref_ts - max(ts)) / 86400"),
    ("user", "user_active_days", "float", "Span of user activity in days", "(max(ts) - min(ts)) / 86400"),
    ("user", "user_preferred_genre", "str", "Genre the user rated most often (>=4 stars weighted)", "argmax over genre counts of liked items"),
    ("user", "user_age", "int", "Age from the user profile", "u.user profile"),
    ("user", "user_gender", "str", "Gender from the user profile (M/F)", "u.user profile"),
    ("user", "user_occupation", "str", "Occupation from the user profile", "u.user profile"),
    ("user", "user_age_group", "str", "Age bucket (<18 ... 65+)", "binned in preparation"),
    ("item", "item_n_ratings", "int", "Popularity: number of train ratings for the item", "count over train window"),
    ("item", "item_avg_rating", "float", "Mean rating of the item", "mean(rating) over train window"),
    ("item", "item_rating_std", "float", "Spread of the item's ratings (0 if single rating)", "std(rating), filled with 0"),
    ("item", "item_bayesian_rating", "float", "Damped mean rating; shrinks low-count items toward the global mean", "(n*mean + m*global)/(n+m), m=20"),
    ("item", "item_pop_rank", "int", "Dense popularity rank (1 = most rated)", "rank of item_n_ratings"),
    ("item", "item_days_since_last", "float", "Recency: days since the item was last rated in train", "(ref_ts - max(ts)) / 86400"),
    ("item", "item_release_year", "int", "Release year parsed from release_date / title", "preparation stage"),
    ("item", "item_age_years", "float", "Age of the movie at the reference date", "ref_year - release_year"),
    ("item", "item_genres", "str", "Pipe-joined genre list", "one-hot flags from u.item"),
    ("interaction", "rating_dev_user", "float", "Rating minus the user's average (user bias removed)", "rating - user_avg_rating"),
    ("interaction", "rating_dev_item", "float", "Rating minus the item's average (item bias removed)", "rating - item_avg_rating"),
    ("interaction", "event_hour", "int", "Hour of day of the interaction (UTC)", "from event_time"),
    ("interaction", "event_dow", "int", "Day of week of the interaction (0=Mon)", "from event_time"),
    ("interaction", "is_weekend", "int", "1 if the interaction happened on Sat/Sun", "event_dow >= 5"),
]


def _preferred_genre(train: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """Most-rated genre among the items each user liked (rating >= 4)."""
    liked = train[train["rating"] >= 4].merge(
        items[["movie_id"] + GENRES], on="movie_id", how="left")
    counts = liked.groupby("user_id")[GENRES].sum()
    pref = counts.idxmax(axis=1)
    return pref


def run() -> dict:
    cfg = load_config()
    logger = get_logger("features", cfg["paths"]["logs"])
    proc, feat = cfg["paths"]["data_processed"], cfg["paths"]["data_features"]

    train = pd.read_parquet(proc / "train.parquet")
    users = pd.read_parquet(proc / "users.parquet")
    items = pd.read_parquet(proc / "items.parquet")
    ua = pd.read_parquet(proc / "user_activity.parquet")
    ip = pd.read_parquet(proc / "item_popularity.parquet")

    ref_ts = int(train["timestamp"].max())  # "as of" moment for all features
    ref_year = pd.to_datetime(ref_ts, unit="s", utc=True).year

    # ---- user features ----------------------------------------------
    uf = ua.rename(columns={
        "n_interactions": "user_n_ratings", "avg_rating": "user_avg_rating",
        "rating_std": "user_rating_std", "active_days": "user_active_days"})
    uf["user_rating_std"] = uf["user_rating_std"].fillna(0.0)
    uf["user_days_since_last"] = ((ref_ts - uf["last_ts"]) / 86400).round(2)
    uf = uf.drop(columns=["first_ts", "last_ts"])
    uf = uf.merge(_preferred_genre(train, items).rename("user_preferred_genre"),
                  on="user_id", how="left")
    uf["user_preferred_genre"] = uf["user_preferred_genre"].fillna("unknown")
    uf = uf.merge(users.rename(columns={
        "age": "user_age", "gender": "user_gender",
        "occupation": "user_occupation", "age_group": "user_age_group"})
        [["user_id", "user_age", "user_gender", "user_occupation", "user_age_group"]],
        on="user_id", how="left")
    uf.to_parquet(feat / "user_features.parquet", index=False)

    # ---- item features ----------------------------------------------
    itf = ip.rename(columns={
        "n_ratings": "item_n_ratings", "avg_rating": "item_avg_rating",
        "rating_std": "item_rating_std", "bayesian_rating": "item_bayesian_rating",
        "pop_rank": "item_pop_rank"})
    itf["item_rating_std"] = itf["item_rating_std"].fillna(0.0)
    itf["item_days_since_last"] = ((ref_ts - itf["last_ts"]) / 86400).round(2)
    itf = itf.drop(columns=["last_ts"])
    itf = itf.merge(items[["movie_id", "release_year", "genres"]], on="movie_id", how="left")
    itf = itf.rename(columns={"release_year": "item_release_year", "genres": "item_genres"})
    itf["item_age_years"] = (ref_year - itf["item_release_year"]).astype("Float64")
    itf.to_parquet(feat / "item_features.parquet", index=False)

    # ---- interaction features (train) --------------------------------
    inf = train.merge(uf[["user_id", "user_avg_rating"]], on="user_id", how="left")
    inf = inf.merge(itf[["movie_id", "item_avg_rating"]], on="movie_id", how="left")
    inf["rating_dev_user"] = inf["rating"] - inf["user_avg_rating"]
    inf["rating_dev_item"] = inf["rating"] - inf["item_avg_rating"]
    et = pd.to_datetime(inf["timestamp"], unit="s", utc=True)
    inf["event_hour"] = et.dt.hour
    inf["event_dow"] = et.dt.dayofweek
    inf["is_weekend"] = (inf["event_dow"] >= 5).astype(int)
    keep = ["user_id", "movie_id", "rating", "timestamp", "rating_dev_user",
            "rating_dev_item", "event_hour", "event_dow", "is_weekend"]
    inf[keep].to_parquet(feat / "interaction_features.parquet", index=False)

    # ---- generated feature catalog -----------------------------------
    cat = pd.DataFrame(FEATURE_CATALOG,
                       columns=["entity", "feature", "dtype", "description", "computation"])
    cat.to_csv(cfg["paths"]["reports"] / "feature_catalog.csv", index=False)
    md = ["# Feature Catalog", "", "Roll No: 2025AG05429 | DM4ML Assignment 2", "",
          f"All features are computed on the TRAIN window only "
          f"(reference timestamp = {ref_ts}, last train event). "
          "They are served from the Parquet feature store (`data/features/`) through "
          "`src/features/store.py` for BOTH training and prediction.", "",
          "| Entity | Feature | Type | Description | Computation |", "|---|---|---|---|---|"]
    for _, r in cat.iterrows():
        md.append(f"| {r['entity']} | `{r['feature']}` | {r['dtype']} | "
                  f"{r['description']} | {r['computation']} |")
    (cfg["paths"]["reports"] / "feature_catalog.md").write_text("\n".join(md) + "\n",
                                                               encoding="utf-8")

    summary = {"stage": "features", "executed_at_utc": utc_now_iso(),
               "reference_timestamp": ref_ts,
               "user_features": {"rows": len(uf), "columns": list(uf.columns)},
               "item_features": {"rows": len(itf), "columns": list(itf.columns)},
               "interaction_features": {"rows": len(inf), "columns": keep},
               "catalog_entries": len(cat)}
    write_json(summary, cfg["paths"]["reports"] / "features_summary.json")
    logger.info("Feature store written: %d user, %d item, %d interaction rows; "
                "%d features documented in catalog", len(uf), len(itf), len(inf), len(cat))
    return summary


if __name__ == "__main__":
    run()
