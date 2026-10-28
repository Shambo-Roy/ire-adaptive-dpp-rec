"""Online category-interest estimators (DECISIONS.md D-11).

Each estimator turns the observed category sequence c_1..c_T into estimates
z_hat_0..z_hat_T, where z_hat_t uses only c_1..c_t (no look-ahead) and
z_hat_0 = uniform (empty history). Observations are one-hot vectors x_t = e_{c_t}.

    cumulative mean   z_hat_t = z_hat_{t-1} + (x_t - z_hat_{t-1}) / t     (non-adaptive)
    EWMA              z_hat_t = (1 - alpha) z_hat_{t-1} + alpha x_t       (proposed)
    sliding window    z_hat_t = mean of x_s over the last min(t, W) steps

All functions are batched: ``cats`` has shape (B, T) (or (T,)) and the result has shape
(B, T + 1, C) (or (T + 1, C)); row t of the result is z_hat_t.
Every estimate is a probability vector: a convex combination of one-hot vectors and
the uniform starting point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def categories_of(items: np.ndarray, item_categories: np.ndarray) -> np.ndarray:
    """Map observed item ids to their (public) categories."""
    return item_categories[items]


def _prepare(cats: np.ndarray, n_categories: int) -> tuple[np.ndarray, bool]:
    cats = np.asarray(cats)
    single = cats.ndim == 1
    cats = np.atleast_2d(cats)
    if cats.size and (cats.min() < 0 or cats.max() >= n_categories):
        raise ValueError("category index out of range")
    return cats, single


def _one_hot(cats: np.ndarray, n_categories: int) -> np.ndarray:
    return np.eye(n_categories)[cats]  # (B, T, C)


def _finish(z: np.ndarray, single: bool) -> np.ndarray:
    return z[0] if single else z


def cumulative_mean(cats: np.ndarray, n_categories: int) -> np.ndarray:
    """Average of all observations so far; z_hat_0 = uniform."""
    cats, single = _prepare(cats, n_categories)
    B, T = cats.shape
    x = _one_hot(cats, n_categories)
    out = np.empty((B, T + 1, n_categories))
    out[:, 0] = 1.0 / n_categories
    out[:, 1:] = np.cumsum(x, axis=1) / np.arange(1, T + 1)[None, :, None]
    return _finish(out, single)


def ewma(cats: np.ndarray, n_categories: int, alpha: float, z0: np.ndarray | None = None) -> np.ndarray:
    """Exponentially weighted moving average with smoothing factor alpha in (0, 1]."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    cats, single = _prepare(cats, n_categories)
    B, T = cats.shape
    x = _one_hot(cats, n_categories)
    out = np.empty((B, T + 1, n_categories))
    out[:, 0] = np.full(n_categories, 1.0 / n_categories) if z0 is None else z0
    for t in range(1, T + 1):
        out[:, t] = (1.0 - alpha) * out[:, t - 1] + alpha * x[:, t - 1]
    return _finish(out, single)


def sliding_window(cats: np.ndarray, n_categories: int, window: int) -> np.ndarray:
    """Category frequencies over the last min(t, W) observations; z_hat_0 = uniform."""
    if window < 1:
        raise ValueError("window must be >= 1")
    cats, single = _prepare(cats, n_categories)
    B, T = cats.shape
    x = _one_hot(cats, n_categories)
    csum = np.concatenate([np.zeros((B, 1, n_categories)), np.cumsum(x, axis=1)], axis=1)  # csum[:, t] = sum of x_1..x_t
    t = np.arange(T + 1)
    lo = np.maximum(t - window, 0)
    counts = csum[:, t] - csum[:, lo]
    n = (t - lo).astype(float)
    out = np.empty((B, T + 1, n_categories))
    out[:, 0] = 1.0 / n_categories
    out[:, 1:] = counts[:, 1:] / n[None, 1:, None]
    return _finish(out, single)


def ewma_mean_age(alpha: float) -> float:
    """Mean age (in steps) of the observations weighted by EWMA: (1 - alpha) / alpha."""
    return (1.0 - alpha) / alpha


def matched_window(alpha: float) -> int:
    """Window with the same mean observation age as EWMA(alpha): W = 2/alpha - 1 (rounded)."""
    return max(1, int(round(2.0 / alpha - 1.0)))


@dataclass(frozen=True)
class EstimatorSpec:
    """A named estimator configuration, usable as ``spec(cats, C)``."""

    kind: str          # "cumulative" | "ewma" | "window"
    param: float | int | None = None

    @property
    def name(self) -> str:
        if self.kind == "cumulative":
            return "cumulative"
        return f"{self.kind}({self.param})"

    def __call__(self, cats: np.ndarray, n_categories: int) -> np.ndarray:
        if self.kind == "cumulative":
            return cumulative_mean(cats, n_categories)
        if self.kind == "ewma":
            return ewma(cats, n_categories, float(self.param))
        if self.kind == "window":
            return sliding_window(cats, n_categories, int(self.param))
        raise ValueError(f"unknown estimator kind {self.kind!r}")
