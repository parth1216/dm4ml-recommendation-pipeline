# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 3b - Data transformation (derived datasets).

Built from the TRAIN window only (so nothing from the test period leaks
into the models):

  * user_activity.parquet    - per-user activity summary
  * item_popularity.parquet  - per-item counts / mean / Bayesian mean
  * interaction_matrix.npz   - sparse user x item rating matrix (CSR)
  * matrix_mappings.json     - user_id<->row and movie_id<->col indices
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.config import load_config
from src.utils import get_logger, utc_now_iso, write_json


def run() -> dict:
    cfg = load_config()
    logger = get_logger("transformation", cfg["paths"]["logs"])
    proc = cfg["paths"]["data_processed"]

    train = pd.read_parquet(proc / "train.parquet")
    items = pd.read_parquet(proc / "items.parquet")

    # ---- user activity summary ---------------------------------------
    ua = (train.groupby("user_id")
          .agg(n_interactions=("rating", "size"),
               avg_rating=("rating", "mean"),
               rating_std=("rating", "std"),
               first_ts=("timestamp", "min"),
               last_ts=("timestamp", "max"))
          .reset_index())
    ua["active_days"] = ((ua["last_ts"] - ua["first_ts"]) / 86400).round(1)
    ua.to_parquet(proc / "user_activity.parquet", index=False)

    # ---- item popularity ---------------------------------------------
    m = cfg["transformation"]["bayesian_prior_weight"]
    global_mean = float(train["rating"].mean())
    ip = (train.groupby("movie_id")
          .agg(n_ratings=("rating", "size"),
               avg_rating=("rating", "mean"),
               rating_std=("rating", "std"),
               last_ts=("timestamp", "max"))
          .reset_index())
    ip["bayesian_rating"] = ((ip["n_ratings"] * ip["avg_rating"] + m * global_mean)
                             / (ip["n_ratings"] + m))
    ip["pop_rank"] = ip["n_ratings"].rank(method="first", ascending=False).astype(int)
    ip = ip.merge(items[["movie_id", "title"]], on="movie_id", how="left")
    ip.to_parquet(proc / "item_popularity.parquet", index=False)

    # ---- sparse user-item interaction matrix -------------------------
    user_ids = np.sort(train["user_id"].unique())
    movie_ids = np.sort(items["movie_id"].unique())  # full catalog as columns
    u_index = {int(u): i for i, u in enumerate(user_ids)}
    m_index = {int(mv): j for j, mv in enumerate(movie_ids)}
    rows = train["user_id"].map(u_index).to_numpy()
    cols = train["movie_id"].map(m_index).to_numpy()
    mat = sparse.csr_matrix(
        (train["rating"].to_numpy(dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )
    sparse.save_npz(proc / "interaction_matrix.npz", mat)
    write_json({"user_ids": [int(u) for u in user_ids],
                "movie_ids": [int(mv) for mv in movie_ids],
                "global_mean": global_mean},
               proc / "matrix_mappings.json")

    density = mat.nnz / (mat.shape[0] * mat.shape[1])
    summary = {
        "stage": "transformation",
        "executed_at_utc": utc_now_iso(),
        "user_activity_rows": int(len(ua)),
        "item_popularity_rows": int(len(ip)),
        "interaction_matrix": {"shape": list(mat.shape), "nnz": int(mat.nnz),
                               "density": round(float(density), 5)},
        "global_mean_rating": round(global_mean, 4),
    }
    write_json(summary, cfg["paths"]["reports"] / "transformation_summary.json")
    logger.info("Transformation complete: matrix %sx%s nnz=%d density=%.3f%%",
                mat.shape[0], mat.shape[1], mat.nnz, 100 * density)
    return summary


if __name__ == "__main__":
    run()
