# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Stage 5 - Model training.

Trains the two recommenders from the feature store / interaction matrix
and tracks parameters, training statistics and artifacts with MLflow
(file-backed store under ./mlruns). Each model gets its own MLflow run;
the run_id is persisted to models/<name>/metadata.json so that the
evaluation stage can append its metrics to the SAME run.

Usage:  python -m src.training.train --model popularity|item_knn|all
"""
from __future__ import annotations

import argparse
import time

import joblib
import mlflow

from src.config import load_config
from src.features.store import FeatureStore
from src.training.models import ItemKNNRecommender, PopularityRecommender
from src.utils import get_logger, utc_now_iso, write_json


def _setup_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(f"file:{cfg['paths']['mlruns']}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def _train_one(model_key: str, cfg: dict, store: FeatureStore, logger) -> dict:
    item_feats = store.item_features()
    mat, mappings = store.interaction_matrix()

    t0 = time.time()
    if model_key == "popularity":
        params = {"score_feature": cfg["training"]["popularity"]["score_feature"],
                  "bayesian_prior_weight": cfg["transformation"]["bayesian_prior_weight"]}
        model = PopularityRecommender(params["score_feature"])
        model.fit(item_feats, mappings["global_mean"])
    elif model_key == "item_knn":
        knn_cfg = cfg["training"]["item_knn"]
        params = {"k_neighbors": knn_cfg["k_neighbors"],
                  "shrinkage": knn_cfg["shrinkage"],
                  "lambda_item": knn_cfg["lambda_item"],
                  "lambda_user": knn_cfg["lambda_user"],
                  "similarity": "adjusted_cosine",
                  "prediction": "knn_with_baselines"}
        model = ItemKNNRecommender(params["k_neighbors"], params["shrinkage"],
                                   params["lambda_item"], params["lambda_user"])
        model.fit(mat, mappings, item_feats)
    else:
        raise ValueError(f"unknown model '{model_key}'")
    fit_seconds = time.time() - t0

    out_dir = cfg["paths"]["models"] / model.name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")

    with mlflow.start_run(run_name=model.name) as run:
        mlflow.set_tags({"stage": "training", "model_family": model_key,
                         "dataset": cfg["dataset"]["name"],
                         "roll_no": "2025AG05429"})
        mlflow.log_params(params)
        mlflow.log_metrics({
            "fit_seconds": round(fit_seconds, 3),
            "train_users": mat.shape[0],
            "train_items_catalog": mat.shape[1],
            "train_interactions": int(mat.nnz),
        })
        mlflow.log_artifact(str(cfg["paths"]["reports"] / "feature_catalog.csv"))
        mlflow.log_artifact(str(cfg["project_root"] / "configs" / "pipeline.yaml"))
        run_id = run.info.run_id

    meta = {"model": model.name, "trained_at_utc": utc_now_iso(),
            "params": params, "fit_seconds": round(fit_seconds, 3),
            "mlflow_run_id": run_id,
            "model_file": str((out_dir / "model.joblib").relative_to(cfg["project_root"]))}
    write_json(meta, out_dir / "metadata.json")

    sample_user = int(mappings["user_ids"][0])
    logger.info("%s trained in %.2fs (run_id=%s). Sample top-5 for user %d: %s",
                model.name, fit_seconds, run_id, sample_user,
                model.recommend(sample_user, exclude=set(), n=5))
    return meta


def run(which: str = "all") -> list[dict]:
    cfg = load_config()
    logger = get_logger("training", cfg["paths"]["logs"])
    _setup_mlflow(cfg)
    store = FeatureStore(cfg)
    keys = ["popularity", "item_knn"] if which == "all" else [which]
    return [_train_one(k, cfg, store, logger) for k in keys]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["popularity", "item_knn", "all"], default="all")
    run(ap.parse_args().model)
