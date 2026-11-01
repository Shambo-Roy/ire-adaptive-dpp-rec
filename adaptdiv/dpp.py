"""Greedy approximate MAP inference for an interest-weighted DPP (DECISIONS.md D-14, D-15).

Objective over size-K sets Y, for scores s_i > 0 and a positive semidefinite similarity S:

    F(Y) = theta * sum_{i in Y} log s_i + (1 - theta) * log det S_Y,    theta in [0, 1]

For theta < 1 this is the DPP L = Diag(q) S Diag(q) with q_i = s_i^(theta / (2 (1 - theta))),
because log det L_Y = (theta / (1 - theta)) * sum log s_i + log det S_Y, so
F(Y) = (1 - theta) * log det L_Y. Theta = 1 is exactly Top-K.

``greedy_map`` follows Algorithm 1 of Chen, Zhang & Zhou (NeurIPS 2018), run on S with the
log-score gain. For each item it maintains c_i (the new Cholesky row) and d_i^2 (the
factor by which adding i multiplies det S_Y):

    det S_{Y + i} = det S_Y * d_i^2

Greedy is an approximation: exact MAP is NP-hard, even to approximate within 8/9 + eps.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

D2_FLOOR = 1e-12  # guards log(0) and division when an item is (nearly) in the span of Y


def _check(log_scores: np.ndarray, S: np.ndarray, theta: float, k: int) -> None:
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if not 1 <= k <= log_scores.shape[0]:
        raise ValueError("k must be between 1 and the number of items")
    if S.shape != (log_scores.shape[0], log_scores.shape[0]):
        raise ValueError("S must be square and match the number of scores")


def greedy_map(log_scores: np.ndarray, S: np.ndarray, theta: float, k: int) -> np.ndarray:
    """Return k distinct item indices in selection order (cost O(k^2 N))."""
    _check(log_scores, S, theta, k)
    N = log_scores.shape[0]
    d2 = np.diag(S).astype(float).copy()
    c = np.empty((N, k - 1))
    available = np.ones(N, dtype=bool)
    picked = np.empty(k, dtype=int)
    for m in range(k):
        gain = theta * log_scores + (1.0 - theta) * np.log(np.maximum(d2, D2_FLOOR))
        gain[~available] = -np.inf
        j = int(np.argmax(gain))
        picked[m] = j
        available[j] = False
        if m < k - 1:
            e = (S[j] - c[:, :m] @ c[j, :m]) / np.sqrt(max(d2[j], D2_FLOOR))
            c[:, m] = e
            d2 = d2 - e * e
    return picked


def objective(items: np.ndarray, log_scores: np.ndarray, S: np.ndarray, theta: float) -> float:
    """F(Y) evaluated directly from the determinant."""
    items = np.asarray(items)
    sign, logdet = np.linalg.slogdet(S[np.ix_(items, items)])
    if sign <= 0:
        return -np.inf if theta < 1.0 else theta * float(log_scores[items].sum())
    return theta * float(log_scores[items].sum()) + (1.0 - theta) * float(logdet)


def naive_greedy_map(log_scores: np.ndarray, S: np.ndarray, theta: float, k: int) -> np.ndarray:
    """Reference greedy recomputing determinants from scratch (test oracle; slow)."""
    _check(log_scores, S, theta, k)
    picked: list[int] = []
    for _ in range(k):
        rest = [i for i in range(log_scores.shape[0]) if i not in picked]
        values = [objective(np.array(picked + [i]), log_scores, S, theta) for i in rest]
        picked.append(rest[int(np.argmax(values))])
    return np.array(picked)


def brute_force_map(log_scores: np.ndarray, S: np.ndarray, theta: float, k: int) -> np.ndarray:
    """Exact maximiser by enumerating all size-k sets (tiny instances only)."""
    _check(log_scores, S, theta, k)
    best = max(combinations(range(log_scores.shape[0]), k),
               key=lambda Y: objective(np.array(Y), log_scores, S, theta))
    return np.array(best)


def dpp_kernel(scores: np.ndarray, S: np.ndarray, theta: float) -> np.ndarray:
    """Explicit kernel L = Diag(q) S Diag(q) for theta < 1 (validation only; may be badly scaled)."""
    if not 0.0 <= theta < 1.0:
        raise ValueError("the kernel form needs theta in [0, 1)")
    q = scores ** (theta / (2.0 * (1.0 - theta)))
    return q[:, None] * S * q[None, :]
