# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Recommender models.

* :class:`PopularityRecommender` - non-personalised baseline that ranks
  items by their Bayesian (damped) mean rating read from the FEATURE
  STORE, so the exact same feature is reused at training and serving.

* :class:`ItemKNNRecommender` - personalised item-based collaborative
  filtering with adjusted-cosine similarity (user-mean-centred ratings),
  significance shrinkage and top-k neighbour pruning.

Both expose the same API:
    fit(...), recommend(user_id, exclude, n) -> [movie_id, ...],
    predict(user_id, movie_id) -> float rating estimate
and are therefore interchangeable inside the evaluation stage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class PopularityRecommender:
    """Baseline: everyone gets the globally best-rated (damped) items."""

    name = "popularity_baseline"

    def __init__(self, score_feature: str = "item_bayesian_rating"):
        self.score_feature = score_feature
        self.ranked_items: list[int] = []
        self.item_scores: dict[int, float] = {}
        self.item_means: dict[int, float] = {}
        self.global_mean: float = 0.0

    def fit(self, item_features: pd.DataFrame, global_mean: float):
        """item_features: feature-store table indexed by movie_id."""
        ranked = item_features.sort_values(self.score_feature, ascending=False)
        self.ranked_items = [int(i) for i in ranked.index]
        self.item_scores = ranked[self.score_feature].astype(float).to_dict()
        self.item_means = item_features["item_avg_rating"].astype(float).to_dict()
        self.global_mean = float(global_mean)
        return self

    def recommend(self, user_id: int, exclude: set[int], n: int) -> list[int]:
        recs = []
        for mid in self.ranked_items:
            if mid not in exclude:
                recs.append(mid)
                if len(recs) == n:
                    break
        return recs

    def predict(self, user_id: int, movie_id: int) -> float:
        return float(self.item_means.get(movie_id, self.global_mean))


class ItemKNNRecommender:
    """Item-based CF with adjusted-cosine similarity and baseline biases.

    Similarity:
        sim(i, j) = cos(r_*i - mean_u, r_*j - mean_u) * n_ij / (n_ij + shrinkage)
    where n_ij is the number of users who rated both items; only the
    top-k most similar neighbours per item are kept.

    Rating prediction (kNN with baselines, Netflix-prize style):
        b(u, i)   = mu + b_u + b_i          (damped user/item biases)
        r_hat(u,i) = b(u,i) + sum_j s_ij * (r_uj - b(u,j)) / sum_j |s_ij|
    """

    name = "item_knn_cf"

    def __init__(self, k_neighbors: int = 30, shrinkage: float = 100.0,
                 lambda_item: float = 10.0, lambda_user: float = 15.0):
        self.k = k_neighbors
        self.shrinkage = shrinkage
        self.lambda_item = lambda_item
        self.lambda_user = lambda_user
        self.b_i: np.ndarray | None = None
        self.b_u: np.ndarray | None = None
        self.sim: sparse.csr_matrix | None = None  # item x item, top-k pruned
        self.user_index: dict[int, int] = {}
        self.movie_ids: np.ndarray | None = None
        self.movie_index: dict[int, int] = {}
        self.mat: sparse.csr_matrix | None = None          # raw ratings
        self.centered: sparse.csr_matrix | None = None     # user-mean-centred
        self.user_means: np.ndarray | None = None
        self.item_means: dict[int, float] = {}
        self.global_mean: float = 0.0
        self.fallback_ranking: list[int] = []

    def fit(self, mat: sparse.csr_matrix, mappings: dict,
            item_features: pd.DataFrame):
        self.mat = mat.tocsr()
        self.user_index = {int(u): i for i, u in enumerate(mappings["user_ids"])}
        self.movie_ids = np.array(mappings["movie_ids"], dtype=int)
        self.movie_index = {int(m): j for j, m in enumerate(self.movie_ids)}
        self.global_mean = float(mappings["global_mean"])
        self.item_means = item_features["item_avg_rating"].astype(float).to_dict()
        # popularity fallback keeps the recommendation list full for
        # users whose neighbourhood yields fewer than n candidates
        self.fallback_ranking = [int(i) for i in item_features
                                 .sort_values("item_bayesian_rating", ascending=False).index]

        # ---- user-mean centring (nonzero entries only) ---------------
        counts = np.diff(self.mat.indptr)
        sums = np.asarray(self.mat.sum(axis=1)).ravel()
        self.user_means = np.divide(sums, counts, out=np.full_like(sums, self.global_mean),
                                    where=counts > 0)
        centered = self.mat.copy().astype(np.float64)
        centered.data = centered.data - np.repeat(self.user_means, counts)
        self.centered = centered.tocsr()

        # ---- adjusted-cosine similarity with shrinkage ---------------
        X = self.centered
        dot = (X.T @ X).toarray()                      # item x item numerator
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        denom = np.outer(norms, norms) + 1e-9
        sim = dot / denom
        B = self.mat.copy()
        B.data = np.ones_like(B.data)
        co = (B.T @ B).toarray()                       # co-rater counts n_ij
        sim *= np.divide(co, co + self.shrinkage,
                         out=np.zeros_like(sim), where=(co + self.shrinkage) > 0)
        np.fill_diagonal(sim, 0.0)

        # ---- damped baseline biases (for rating prediction) ----------
        mu = self.global_mean
        Rc = self.mat.tocsc()
        n_i = np.diff(Rc.indptr)
        sum_i = np.asarray(Rc.sum(axis=0)).ravel()
        self.b_i = np.divide(sum_i - n_i * mu, self.lambda_item + n_i,
                             out=np.zeros_like(sum_i), where=n_i > 0)
        n_u = counts
        resid = self.mat.copy().astype(np.float64)
        resid.data = resid.data - mu - self.b_i[resid.indices]
        sum_u = np.asarray(resid.sum(axis=1)).ravel()
        self.b_u = np.divide(sum_u, self.lambda_user + n_u,
                             out=np.zeros_like(sum_u), where=n_u > 0)

        # ---- top-k pruning ------------------------------------------
        n_items = sim.shape[0]
        keep = min(self.k, n_items - 1)
        idx = np.argpartition(-np.abs(sim), keep, axis=1)[:, :keep]
        pruned = np.zeros_like(sim)
        rows = np.arange(n_items)[:, None]
        pruned[rows, idx] = sim[rows, idx]
        self.sim = sparse.csr_matrix(pruned.astype(np.float32))
        return self

    # ------------------------------------------------------------------
    def recommend(self, user_id: int, exclude: set[int], n: int) -> list[int]:
        ui = self.user_index.get(user_id)
        if ui is None:  # cold-start user -> popularity fallback
            return [m for m in self.fallback_ranking if m not in exclude][:n]
        scores = (self.centered[ui] @ self.sim).toarray().ravel()
        order = np.argsort(-scores)
        recs = []
        for j in order:
            if scores[j] <= 0:
                break
            mid = int(self.movie_ids[j])
            if mid not in exclude:
                recs.append(mid)
                if len(recs) == n:
                    return recs
        for mid in self.fallback_ranking:  # fill remaining slots
            if mid not in exclude and mid not in recs:
                recs.append(mid)
                if len(recs) == n:
                    break
        return recs

    def predict(self, user_id: int, movie_id: int) -> float:
        mu = self.global_mean
        ui = self.user_index.get(user_id)
        mj = self.movie_index.get(movie_id)
        if ui is None and mj is None:
            return mu
        if ui is None:  # cold-start user: global mean + item bias
            return float(np.clip(mu + self.b_i[mj], 1.0, 5.0))
        if mj is None:  # unseen item: global mean + user bias
            return float(np.clip(mu + self.b_u[ui], 1.0, 5.0))

        base = mu + self.b_u[ui] + self.b_i[mj]
        sims = np.asarray(self.sim[mj].todense()).ravel()
        user_row = self.mat[ui]
        rated_idx = user_row.indices
        s = sims[rated_idx]
        mask = s > 0  # positive neighbours only
        if mask.any():
            r = user_row.data[mask]
            base_j = mu + self.b_u[ui] + self.b_i[rated_idx[mask]]
            base += float((s[mask] * (r - base_j)).sum()) / float(np.abs(s[mask]).sum())
        return float(np.clip(base, 1.0, 5.0))
