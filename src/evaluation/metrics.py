# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Ranking and rating metrics (pure functions, unit-tested)."""
from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    top = recommended[:k]
    return sum(1 for m in top if m in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    return sum(1 for m in top if m in relevant) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@k."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2)
              for i, m in enumerate(recommended[:k]) if m in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def catalog_coverage(all_recommendations: list[list[int]], catalog_size: int) -> float:
    """Share of the recommendable catalog that appears in at least one list."""
    unique = {m for recs in all_recommendations for m in recs}
    return len(unique) / catalog_size if catalog_size else 0.0


def intra_list_diversity(recommended: list[int],
                         genre_vectors: dict[int, np.ndarray]) -> float:
    """1 - mean pairwise cosine similarity of the items' genre vectors."""
    vecs = [genre_vectors[m] for m in recommended if m in genre_vectors]
    if len(vecs) < 2:
        return 0.0
    V = np.vstack(vecs).astype(float)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = (V / norms) @ (V / norms).T
    n = len(vecs)
    iu = np.triu_indices(n, k=1)
    return float(1.0 - S[iu].mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))
