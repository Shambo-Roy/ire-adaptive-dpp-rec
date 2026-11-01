"""Base relevance and category-weighted scores (DECISIONS.md D-13, D-26).

Inputs are observable only: training-population item counts, public item categories and
interest estimates. Every item in the catalogue is a candidate for every method (D-13);
already-clicked items are not excluded (D-24).

    global popularity          r_glob_i = (count_i + 1) / (sum_j count_j + N)
    within-category popularity r_i      = (count_i + 1) / (sum_{j in c_i} count_j + n_c)
    plug-in score              s_i(t)   = r_i * zhat_{t, c_i}

The plug-in score is an estimate of the next-item probability: because r sums to 1 within
each category and zhat sums to 1 over categories, s(t) sums to 1 over items. The quality
transform g used by the DPP is specified at Milestone 6 (D-14).
"""
from __future__ import annotations

import numpy as np


def global_popularity(counts: np.ndarray) -> np.ndarray:
    """Add-one smoothed share of all training interactions (Popularity Top-K floor only)."""
    return (counts + 1.0) / (counts.sum() + counts.shape[0])


def within_category_popularity(counts: np.ndarray, categories: np.ndarray, n_categories: int) -> np.ndarray:
    """Add-one smoothed share of training interactions within each item's category."""
    totals = np.bincount(categories, weights=counts + 1.0, minlength=n_categories)
    return (counts + 1.0) / totals[categories]


def plugin_scores(zhat: np.ndarray, r: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """r_i * zhat_{c_i} for all items; ``zhat`` has shape (..., C), the result (..., N)."""
    return zhat[..., categories] * r


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores along the last axis, best first (ties: lowest index)."""
    # A stable sort on the negated scores keeps the order deterministic.
    return np.argsort(-scores, axis=-1, kind="stable")[..., :k]
