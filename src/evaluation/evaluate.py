# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 6 - Model evaluation.

Protocol
--------
* For every test-window user who also exists in train, each model
  recommends K=10 items it has NOT seen the user interact with in train.
* An item is *relevant* if the user rated it >= 4.0 in the test window.
* Ranking metrics: Precision@K, Recall@K, NDCG@K (averaged over users
  with at least one relevant test item).
* Catalog Coverage: share of train-rated items that appear in at least
  one recommendation list.
* Diversity: mean intra-list diversity (1 - pairwise genre cosine).
* Rating metrics: RMSE / MAE over all test pairs whose user is known.

Each model's metrics are appended to ITS OWN MLflow training run
(run_id read from models/<name>/metadata.json), and a side-by-side
comparison table + chart is written to reports/evaluation/.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation import metrics as M
from src.ingestion.ingest import GENRES
from src.utils import get_logger, utc_now_iso, write_json

MODEL_DIRS = ["popularity_baseline", "item_knn_cf"]


def _genre_vectors(items: pd.DataFrame) -> dict[int, np.ndarray]:
    return {int(r["movie_id"]): r[GENRES].to_numpy(dtype=float)
            for _, r in items.iterrows()}


def _evaluate_model(model, train, test, items, cfg, logger) -> dict:
    K = cfg["evaluation"]["top_k"]
    rel_thr = cfg["evaluation"]["relevance_threshold"]

    seen = train.groupby("user_id")["movie_id"].agg(set).to_dict()
    test_by_user = test.groupby("user_id")
    gvecs = _genre_vectors(items)
    catalog = set(train["movie_id"].unique())

    precisions, recalls, ndcgs, ilds = [], [], [], []
    all_recs: list[list[int]] = []
    y_true, y_pred = [], []
    n_eval_users = 0

    for user_id, grp in test_by_user:
        if user_id not in seen:       # cold-start user: no train history
            continue
        # ---- rating prediction over every test pair ------------------
        for _, row in grp.iterrows():
            y_true.append(float(row["rating"]))
            y_pred.append(model.predict(int(user_id), int(row["movie_id"])))
        # ---- top-K ranking ------------------------------------------
        relevant = set(grp.loc[grp["rating"] >= rel_thr, "movie_id"].astype(int))
        recs = model.recommend(int(user_id), exclude=seen[user_id], n=K)
        all_recs.append(recs)
        ilds.append(M.intra_list_diversity(recs, gvecs))
        if relevant:
            n_eval_users += 1
            precisions.append(M.precision_at_k(recs, relevant, K))
            recalls.append(M.recall_at_k(recs, relevant, K))
            ndcgs.append(M.ndcg_at_k(recs, relevant, K))

    results = {
        f"precision_at_{K}": float(np.mean(precisions)),
        f"recall_at_{K}": float(np.mean(recalls)),
        f"ndcg_at_{K}": float(np.mean(ndcgs)),
        "coverage": M.catalog_coverage(all_recs, len(catalog)),
        "diversity": float(np.mean(ilds)),
        "rmse": M.rmse(np.array(y_true), np.array(y_pred)),
        "mae": M.mae(np.array(y_true), np.array(y_pred)),
        "n_users_ranked": n_eval_users,
        "n_rating_pairs": len(y_true),
    }
    logger.info("%s -> %s", model.name,
                {k: round(v, 4) for k, v in results.items()})
    return results


def _sample_recommendations(models: dict, train, test, items, cfg) -> pd.DataFrame:
    """Top-K lists for a few real test users, with titles (for the report)."""
    K = cfg["evaluation"]["top_k"]
    n_users = cfg["evaluation"]["sample_users_in_report"]
    titles = items.set_index("movie_id")["title"].to_dict()
    seen = train.groupby("user_id")["movie_id"].agg(set).to_dict()
    active = [u for u in test["user_id"].value_counts().index if u in seen][:n_users]
    rows = []
    for u in active:
        for name, model in models.items():
            recs = model.recommend(int(u), exclude=seen[u], n=K)
            rows.append({"user_id": int(u), "model": name,
                         "top_recommendations":
                             "; ".join(titles.get(m, str(m)) for m in recs[:5])})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    cfg = load_config()
    logger = get_logger("evaluation", cfg["paths"]["logs"])
    proc = cfg["paths"]["data_processed"]
    out_dir = cfg["paths"]["reports"] / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")
    items = pd.read_parquet(proc / "items.parquet")

    mlflow.set_tracking_uri(f"file:{cfg['paths']['mlruns']}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    models, rows = {}, []
    for mdir in MODEL_DIRS:
        meta_file = cfg["paths"]["models"] / mdir / "metadata.json"
        model = joblib.load(cfg["paths"]["models"] / mdir / "model.joblib")
        models[model.name] = model
        res = _evaluate_model(model, train, test, items, cfg, logger)
        with open(meta_file, encoding="utf-8") as fh:
            run_id = json.load(fh)["mlflow_run_id"]
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k: v for k, v in res.items()
                                if isinstance(v, (int, float))})
            mlflow.set_tag("evaluated_at_utc", utc_now_iso())
        rows.append({"model": model.name, **res})

    comp = pd.DataFrame(rows).set_index("model").round(4)
    comp.to_csv(out_dir / "model_comparison.csv")
    md = ["# Model Evaluation Results", "",
          "Roll No: 2025AG05429 | DM4ML Assignment 2", "",
          f"K = {cfg['evaluation']['top_k']}, relevance threshold = "
          f"{cfg['evaluation']['relevance_threshold']} stars, "
          "chronological train/test split.", "",
          comp.to_markdown(), ""]
    samples = _sample_recommendations(models, train, test, items, cfg)
    samples.to_csv(out_dir / "sample_recommendations.csv", index=False)
    md += ["## Sample recommendations (top-5 shown)", "", samples.to_markdown(index=False), ""]
    (out_dir / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    # ---- comparison chart --------------------------------------------
    plot_cols = [c for c in comp.columns
                 if c.split("_at_")[0] in ("precision", "recall", "ndcg")
                 or c in ("coverage", "diversity")]
    ax = comp[plot_cols].T.plot.bar(rot=20, figsize=(9, 4.5),
                                    color=["#8da0cb", "#fc8d62"])
    ax.set_title("Recommendation quality: popularity baseline vs item-kNN CF "
                 "(Roll No 2025AG05429)")
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg["paths"]["reports"] / "figures" / "model_comparison.png", dpi=150)
    plt.close()

    write_json({"stage": "evaluation", "executed_at_utc": utc_now_iso(),
                "results": rows}, out_dir / "evaluation_summary.json")
    logger.info("Evaluation complete - comparison written to %s",
                out_dir / "model_comparison.md")
    return comp


if __name__ == "__main__":
    run()
