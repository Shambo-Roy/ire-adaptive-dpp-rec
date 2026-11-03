"""Calibrated re-ranking after Steck (RecSys 2018), as adapted in DECISIONS.md D-19.

For a target category distribution p (the method's own profile zhat_t) and a size-K list R:

    q_R(g) = |{i in R : c_i = g}| / |R|,      q~_R = (1 - a) q_R + a p,      a = 0.01
    C_KL(p, R) = KL(p || q~_R) = sum_g p_g log(p_g / q~_R(g))

The list maximises (1 - lam) * sum_{i in R} log s_i - lam * C_KL(p, R), built greedily one item at
a time as in the paper (Eq. 6). lam = 0 is Top-K. Our adaptations: log s_i instead of s_i (the
paper allows a monotone transform), no rank weights and no diversity prior (as in its experiments).
"""
from __future__ import annotations

import numpy as np

KL_SMOOTHING = 0.01  # a in q~ = (1 - a) q + a p (Steck's alpha)


def kl_to_smoothed(target: np.ndarray, q: np.ndarray, a: float = KL_SMOOTHING) -> np.ndarray:
    """KL(target || (1 - a) q + a target) over the last axis; terms with target = 0 count as 0."""
    q_s = (1.0 - a) * q + a * target
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(target > 0, target * np.log(target / q_s), 0.0)
    return terms.sum(axis=-1)


def miscalibration(list_categories: np.ndarray, target: np.ndarray, n_categories: int,
                   a: float = KL_SMOOTHING) -> np.ndarray:
    """C_KL of each list against a target distribution; ``list_categories`` (n, K), ``target`` (n, C)."""
    counts = np.zeros((list_categories.shape[0], n_categories))
    np.add.at(counts, (np.arange(list_categories.shape[0])[:, None], list_categories), 1.0)
    return kl_to_smoothed(target, counts / list_categories.shape[1], a)


def calibrated_lists(log_s: np.ndarray, targets: np.ndarray, categories: np.ndarray, lam: float, k: int,
                     a: float = KL_SMOOTHING) -> np.ndarray:
    """Greedy calibrated lists in selection order, batched over rows: ``log_s`` (n, N), ``targets`` (n, C).

    A candidate's calibration term depends only on its category, so each step evaluates the
    C possible KL values and broadcasts them to items.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be in [0, 1]")
    n, N = log_s.shape
    C = targets.shape[1]
    if not 1 <= k <= N:
        raise ValueError("k must be between 1 and the number of items")
    rows = np.arange(n)
    counts = np.zeros((n, C))
    available = np.ones((n, N), dtype=bool)
    picked = np.empty((n, k), dtype=int)
    eye = np.eye(C)
    for m in range(k):
        q_next = (counts[:, None, :] + eye[None]) / (m + 1)              # (n, C, C): list after adding category c
        kl = kl_to_smoothed(targets[:, None, :], q_next, a)              # (n, C)
        gain = (1.0 - lam) * log_s - lam * kl[:, categories]
        gain[~available] = -np.inf
        j = np.argmax(gain, axis=1)
        picked[:, m] = j
        available[rows, j] = False
        counts[rows, categories[j]] += 1.0
    return picked
