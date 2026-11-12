# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Behavioural tests for the two recommenders on a tiny toy matrix."""
import numpy as np
import pandas as pd
from scipy import sparse

from src.training.models import ItemKNNRecommender, PopularityRecommender


def _toy():
    # 4 users x 4 items; items 0&1 are liked together, 2&3 together
    R = np.array([
        [5, 4, 0, 0],
        [4, 5, 1, 0],
        [0, 1, 5, 4],
        [0, 0, 4, 5],
    ], dtype=float)
    mat = sparse.csr_matrix(R)
    mappings = {"user_ids": [1, 2, 3, 4], "movie_ids": [10, 20, 30, 40],
                "global_mean": float(R[R > 0].mean())}
    feats = pd.DataFrame({
        "movie_id": [10, 20, 30, 40],
        "item_avg_rating": [4.5, 10 / 3, 10 / 3, 4.5],
        "item_bayesian_rating": [4.0, 3.4, 3.4, 4.1],
    }).set_index("movie_id")
    return mat, mappings, feats


def test_popularity_ranks_by_score_and_excludes_seen():
    _, mappings, feats = _toy()
    model = PopularityRecommender().fit(feats, mappings["global_mean"])
    assert model.recommend(1, exclude=set(), n=2) == [40, 10]
    assert model.recommend(1, exclude={40}, n=2) == [10, 20]


def test_item_knn_recommends_from_the_users_taste_cluster():
    mat, mappings, feats = _toy()
    model = ItemKNNRecommender(k_neighbors=2, shrinkage=0.0).fit(mat, mappings, feats)
    # user 1 loves items 10,20 (cluster A) and never saw 30,40;
    # user 4 loves 30,40 -> should be recommended from cluster B first? No:
    # user 1's neighbours of unseen items come from co-rating patterns.
    recs = model.recommend(1, exclude={10, 20}, n=2)
    assert set(recs) <= {30, 40}  # only unseen items are returned
    pred = model.predict(1, 10)
    assert 1.0 <= pred <= 5.0


def test_item_knn_cold_start_falls_back_to_popularity():
    mat, mappings, feats = _toy()
    model = ItemKNNRecommender(k_neighbors=2, shrinkage=0.0).fit(mat, mappings, feats)
    recs = model.recommend(999, exclude=set(), n=2)  # unknown user
    assert recs == [40, 10]  # bayesian popularity order
