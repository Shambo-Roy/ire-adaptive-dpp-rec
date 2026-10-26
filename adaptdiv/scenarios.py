"""Latent-interest construction and drift trajectories (SIMULATION_DESIGN.md sections 3-4).

Time convention: steps are t = 1..T; row k of a trajectory array holds z_t for t = k + 1.
"""
from __future__ import annotations

import numpy as np

from .config import SCENARIOS, ScenarioConfig


def sample_base_interest(kappa: float, n_categories: int, rng: np.random.Generator) -> np.ndarray:
    """z_A ~ Dirichlet(kappa * 1_C)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    z = rng.dirichlet(np.full(n_categories, kappa))
    return z / z.sum()  # guard against floating-point drift away from the simplex


def half_swap_permutation(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return ``perm`` mapping category c's mass to category ``perm[c]``.

    The top-C/2 categories of z (by mass) are sent, by a random bijection, onto the
    bottom-C/2 categories and vice versa (D-23). For odd C the middle category stays put.
    """
    C = z.shape[0]
    order = np.argsort(-z, kind="stable")  # categories by decreasing mass
    h = C // 2
    top, bottom = order[:h], order[C - h:]
    perm = np.arange(C)
    perm[top] = rng.permutation(bottom)
    perm[bottom] = rng.permutation(top)
    return perm


def apply_permutation(z: np.ndarray, perm: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    out[perm] = z
    return out


def shifted_interest(z_a: np.ndarray, perm: np.ndarray, magnitude: float = 1.0) -> np.ndarray:
    """z_B(m) = (1 - m) z_A + m * permuted(z_A)."""
    if not 0.0 <= magnitude <= 1.0:
        raise ValueError("shift magnitude must be in [0, 1]")
    return (1.0 - magnitude) * z_a + magnitude * apply_permutation(z_a, perm)


def mixing_weights(name: str, cfg: ScenarioConfig) -> np.ndarray:
    """Weight w_t on z_B for t = 1..T, so z_t = (1 - w_t) z_A + w_t z_B."""
    t = np.arange(1, cfg.horizon + 1)
    if name in ("stable", "noisy"):
        return np.zeros(cfg.horizon)
    if name == "abrupt":
        return (t >= cfg.abrupt_switch).astype(float)
    if name == "gradual":
        return np.clip((t - cfg.gradual_start) / cfg.gradual_width, 0.0, 1.0)
    if name == "recurrent":
        t1, t2 = cfg.recurrent_switches
        return ((t >= t1) & (t < t2)).astype(float)
    raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")


def trajectory(name: str, z_a: np.ndarray, z_b: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
    """(T, C) array of z_t. Convex combinations keep every row on the simplex."""
    w = mixing_weights(name, cfg)[:, None]
    return (1.0 - w) * z_a[None, :] + w * z_b[None, :]


def noise_rate(name: str, base_rate: float, cfg: ScenarioConfig) -> float:
    """eta for a scenario: the high rate in "noisy", the base rate elsewhere."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}")
    return cfg.noisy_noise_rate if name == "noisy" else base_rate
