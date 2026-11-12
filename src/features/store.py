# Roll No: 2025AG05429 | DM4ML Assignment 2
"""A minimal Parquet-backed feature store.

The same class is used at TRAINING time (models consume feature tables)
and at PREDICTION time (look up a single user's / item's feature vector),
which is what makes the features reusable across both phases.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import sparse

from src.config import load_config


class FeatureStore:
    """Read-side API over data/features/ and the interaction matrix."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._feat: Path = self.cfg["paths"]["data_features"]
        self._proc: Path = self.cfg["paths"]["data_processed"]
        self._cache: dict[str, pd.DataFrame] = {}

    # ---- table-level access (training) -------------------------------
    def user_features(self) -> pd.DataFrame:
        return self._load("user_features", index="user_id")

    def item_features(self) -> pd.DataFrame:
        return self._load("item_features", index="movie_id")

    def interaction_features(self) -> pd.DataFrame:
        return self._load("interaction_features")

    # ---- row-level access (prediction / serving) ---------------------
    def get_user(self, user_id: int) -> pd.Series | None:
        df = self.user_features()
        return df.loc[user_id] if user_id in df.index else None

    def get_item(self, movie_id: int) -> pd.Series | None:
        df = self.item_features()
        return df.loc[movie_id] if movie_id in df.index else None

    # ---- interaction matrix (collaborative filtering) ----------------
    def interaction_matrix(self):
        mat = sparse.load_npz(self._proc / "interaction_matrix.npz")
        with open(self._proc / "matrix_mappings.json", encoding="utf-8") as fh:
            mappings = json.load(fh)
        return mat, mappings

    def _load(self, name: str, index: str | None = None) -> pd.DataFrame:
        if name not in self._cache:
            df = pd.read_parquet(self._feat / f"{name}.parquet")
            if index:
                df = df.set_index(index)
            self._cache[name] = df
        return self._cache[name]
