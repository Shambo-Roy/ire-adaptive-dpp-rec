"""Paired statistics for the confirmatory analysis (DECISIONS.md D-21)."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def wilcoxon_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired differences. All-zero differences give 1."""
    diff = np.asarray(diff, dtype=float)
    if np.allclose(diff, 0.0):
        return 1.0
    return float(sps.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)


def bootstrap_ci(diff: np.ndarray, seed: int = 0, n_boot: int = 10_000, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean paired difference (resampling seeds)."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.shape[0], (n_boot, diff.shape[0]))].mean(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    m = p.shape[0]
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
