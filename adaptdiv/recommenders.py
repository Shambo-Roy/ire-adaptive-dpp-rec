"""Recommendation methods (DECISIONS.md D-14, D-16).

A method combines a profile source with a selector:

    profile:  none (popularity) | static | adaptive | oracle
    selector: Top-K, or greedy DPP with trade-off theta

Scores are s_i = r_i * (zhat_{c_i} + eps); Top-K is the theta = 1 case of the same objective.
The list for step t + 1 uses the profile row for time t only. Profiles are computed
outside this module: ``profiles`` below is whatever the caller passes, and only
experiment code may pass true interests, for the labelled oracle diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dpp import greedy_map
from .relevance import top_k

SCORE_EPS = 0.01  # epsilon in s_i: keeps quality positive when zhat_c = 0 (D-14)
SELECTED_THETA = 0.5  # chosen on tuning seeds under the D-29 rule (results/m7_theta/)


@dataclass(frozen=True)
class Method:
    name: str
    profile: str          # "none" | "static" | "adaptive" | "oracle"
    theta: float = 1.0    # 1.0 = Top-K


def standard_methods(theta: float) -> tuple[Method, ...]:
    """The D-16 method set for a given DPP trade-off."""
    return (
        Method("popularity", "none"),
        Method("static_topk", "static"),
        Method("adaptive_topk", "adaptive"),
        Method("static_dpp", "static", theta),
        Method("adaptive_dpp", "adaptive", theta),
        Method("oracle_topk", "oracle"),
        Method("oracle_dpp", "oracle", theta),
    )


def log_scores(profile: np.ndarray, r: np.ndarray, categories: np.ndarray, eps: float = SCORE_EPS) -> np.ndarray:
    """log s_i = log r_i + log(zhat_{c_i} + eps); ``profile`` has shape (..., C)."""
    return np.log(r) + np.log(profile[..., categories] + eps)


def select(log_s: np.ndarray, S: np.ndarray, theta: float, k: int) -> np.ndarray:
    """Top-K for theta = 1, otherwise greedy DPP. Always returns k distinct items."""
    return top_k(log_s, k) if theta >= 1.0 else greedy_map(log_s, S, theta, k)


def recommendation_lists(
    profiles: np.ndarray,
    r: np.ndarray,
    categories: np.ndarray,
    S: np.ndarray,
    theta: float,
    k: int,
    steps: np.ndarray | None = None,
) -> np.ndarray:
    """Lists for one user: row n holds the list built from ``profiles[steps[n]]``, shape (len(steps), k)."""
    steps = np.arange(profiles.shape[0]) if steps is None else steps
    log_s = log_scores(profiles[steps], r, categories)
    if theta >= 1.0:
        return top_k(log_s, k)
    return np.stack([greedy_map(row, S, theta, k) for row in log_s])


def popularity_list(global_r: np.ndarray, k: int) -> np.ndarray:
    """The fixed Popularity Top-K list (no profile)."""
    return top_k(global_r, k)
