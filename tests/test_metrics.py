# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Hand-computed sanity checks for the evaluation metrics."""
import numpy as np

from src.evaluation import metrics as M


def test_precision_recall():
    recs, rel = [1, 2, 3, 4, 5], {2, 5, 9}
    assert M.precision_at_k(recs, rel, 5) == 2 / 5
    assert M.recall_at_k(recs, rel, 5) == 2 / 3


def test_ndcg_perfect_and_worst():
    assert M.ndcg_at_k([1, 2], {1, 2}, 2) == 1.0
    assert M.ndcg_at_k([3, 4], {1, 2}, 2) == 0.0
    # relevant item at rank 2 only: dcg = 1/log2(3), idcg = 1/log2(2)
    expected = (1 / np.log2(3)) / 1.0
    assert abs(M.ndcg_at_k([9, 1], {1}, 2) - expected) < 1e-9


def test_coverage_and_diversity():
    assert M.catalog_coverage([[1, 2], [2, 3]], 10) == 0.3
    gv = {1: np.array([1, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])}
    assert M.intra_list_diversity([1, 2], gv) == 0.0          # identical genres
    assert abs(M.intra_list_diversity([1, 3], gv) - 1.0) < 1e-9  # orthogonal


def test_rmse_mae():
    t, p = np.array([4.0, 2.0]), np.array([3.0, 2.0])
    assert abs(M.rmse(t, p) - np.sqrt(0.5)) < 1e-9
    assert abs(M.mae(t, p) - 0.5) < 1e-9
