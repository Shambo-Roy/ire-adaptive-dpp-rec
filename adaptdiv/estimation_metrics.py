"""Interest-estimation metrics (PROVISIONAL: formal definitions are locked at M7, D-17).

Alignment convention: ``z`` has shape (T, C) with row k = z_t for t = k + 1; ``zhat`` has
shape (T + 1, C) with row t = estimate after observing c_1..c_t. The *tracking error* at
step t compares zhat_t with z_t (the interest that generated the t-th observation).
All functions also accept a leading batch axis.
"""
from __future__ import annotations

import numpy as np


def tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Total-variation distance 0.5 * ||a - b||_1 over the last axis."""
    return 0.5 * np.abs(a - b).sum(axis=-1)


def tracking_error(zhat: np.ndarray, z: np.ndarray) -> np.ndarray:
    """TV(zhat_t, z_t) for t = 1..T; shape (..., T)."""
    return tv(zhat[..., 1:, :], z)


def noise_floor(z: np.ndarray, eta: float) -> np.ndarray:
    """TV between the observed-category distribution (1-eta) z + eta/C and z: eta * TV(z, uniform)."""
    C = z.shape[-1]
    return eta * tv(z, np.full(C, 1.0 / C))


def crossover_lag(
    zhat: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    start: int,
    run: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Steps after ``start`` until the estimate is closer to ``new`` than to ``old`` for ``run`` consecutive steps.

    ``start`` is the first step t at which the true interest is at least as close to ``new`` as to
    ``old`` (abrupt: tau; gradual: the ramp midpoint). Returns (lag, censored) where lag = L >= 0 is
    the smallest value with TV(zhat_t, new) < TV(zhat_t, old) for t = start+L .. start+L+run-1.
    If that never happens within the horizon, lag = T - start + 1 and censored = True.
    ``old``/``new`` have shape (..., C); ``zhat`` has shape (..., T+1, C).
    """
    closer = tv(zhat, new[..., None, :]) < tv(zhat, old[..., None, :])  # (..., T+1)
    T = zhat.shape[-2] - 1
    seg = closer[..., start:]  # steps start..T
    n = seg.shape[-1]
    lag = np.full(seg.shape[:-1], T - start + 1, dtype=int)
    censored = np.ones(seg.shape[:-1], dtype=bool)
    if n >= run:
        # ok[..., L] is True when seg[L : L + run] are all True
        csum = np.concatenate([np.zeros(seg.shape[:-1] + (1,), dtype=int), np.cumsum(seg, axis=-1)], axis=-1)
        ok = (csum[..., run:] - csum[..., :-run]) == run  # (..., n - run + 1)
        has = ok.any(axis=-1)
        first = ok.argmax(axis=-1)
        lag = np.where(has, first, lag)
        censored = ~has
    return lag, censored


def false_switch_rate(zhat: np.ndarray, z: np.ndarray, burn_in: int = 20) -> np.ndarray:
    """Fraction of steps t > burn_in where argmax zhat changes while argmax z does not.

    Compares t with t - 1 for t = burn_in + 1 .. T (estimates after t vs t-1 observations).
    """
    est_top = zhat[..., 1:, :].argmax(axis=-1)  # (..., T), row k = after k+1 observations
    true_top = z.argmax(axis=-1)
    est_change = est_top[..., 1:] != est_top[..., :-1]
    true_same = true_top[..., 1:] == true_top[..., :-1]
    mask = np.arange(1, z.shape[-2]) >= burn_in  # transitions into step t = k+2 > burn_in
    events = (est_change & true_same)[..., mask]
    opportunities = true_same[..., mask].sum(axis=-1)
    return events.sum(axis=-1) / np.maximum(opportunities, 1)
