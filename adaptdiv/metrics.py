"""Recommendation metrics (DECISIONS.md D-17). Pure functions, batched over lists.

``lists`` has shape (n, K), one list per scored step, sorted by score. ``targets`` has shape
(n, h): the next h clicks after each list. Repeated clicks in a target window count once.
"""
from __future__ import annotations

import numpy as np


def target_matrix(items: np.ndarray, steps: np.ndarray, h: int) -> np.ndarray:
    """Row n holds i_{t+1}..i_{t+h} (0-based items[t : t+h]) for t = steps[n]."""
    return np.stack([items[t: t + h] for t in steps])


def _distinct_mask(targets: np.ndarray) -> np.ndarray:
    """True at the first occurrence of each value in a row, in sorted order alignment."""
    order = np.argsort(targets, axis=1, kind="stable")
    sorted_t = np.take_along_axis(targets, order, axis=1)
    first = np.ones_like(sorted_t, dtype=bool)
    first[:, 1:] = sorted_t[:, 1:] != sorted_t[:, :-1]
    mask = np.zeros_like(first)
    np.put_along_axis(mask, order, first, axis=1)
    return mask


def hits(lists: np.ndarray, targets: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """(n, K) bool: is each listed item a target? ``valid`` (n, h) can disable some targets."""
    match = lists[:, :, None] == targets[:, None, :]
    if valid is not None:
        match = match & valid[:, None, :]
    return match.any(axis=2)


def n_targets(targets: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Number of distinct (valid) targets per row."""
    distinct = _distinct_mask(targets)
    return (distinct if valid is None else distinct & valid).sum(axis=1)


def ndcg_at_k(hit: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """Binary NDCG: DCG = sum hit(p) / log2(p + 1); ideal has min(n_relevant, K) hits. NaN if no targets."""
    K = hit.shape[1]
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    ideal_cum = np.concatenate([[0.0], np.cumsum(disc)])
    ideal = ideal_cum[np.minimum(n_relevant, K)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ideal > 0, (hit * disc).sum(axis=1) / ideal, np.nan)


def recall_at_k(hit: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """Fraction of distinct targets in the list. NaN if no targets."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_relevant > 0, hit.sum(axis=1) / n_relevant, np.nan)


def expected_utility(probs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Sum of true next-click probabilities over the list; ``probs`` is (n, K)."""
    return (probs if mask is None else probs * mask).sum(axis=1)


def category_coverage(list_categories: np.ndarray, n_categories: int) -> np.ndarray:
    """Distinct categories in each list, divided by min(K, C)."""
    K = list_categories.shape[1]
    s = np.sort(list_categories, axis=1)
    distinct = 1 + (s[:, 1:] != s[:, :-1]).sum(axis=1)
    return distinct / min(K, n_categories)


def intra_list_diversity(list_embeddings: np.ndarray) -> np.ndarray:
    """Mean over ordered pairs of 1 - cosine, for unit-norm embeddings (n, K, d)."""
    K = list_embeddings.shape[1]
    sims = np.einsum("nid,njd->nij", list_embeddings, list_embeddings)
    return 1.0 - (sims.sum(axis=(1, 2)) - np.trace(sims, axis1=1, axis2=2)) / (K * (K - 1))
