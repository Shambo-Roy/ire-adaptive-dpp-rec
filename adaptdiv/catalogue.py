"""Synthetic item catalogue (SIMULATION_DESIGN.md section 2).

Items are stored grouped by category: items ``c*n_c .. (c+1)*n_c - 1`` belong to category c.

Two views exist:
- ``Catalogue``       full catalogue incl. true popularity (simulator/evaluation only).
- ``PublicCatalogue`` what estimators/recommenders may see: categories + embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CatalogueConfig


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@dataclass(frozen=True)
class PublicCatalogue:
    """Observable item metadata (leakage firewall, SIMULATION_DESIGN.md section 6)."""

    categories: np.ndarray   # (N,) int, category of each item
    embeddings: np.ndarray   # (N, d) float, unit-norm item embeddings
    n_categories: int


@dataclass(frozen=True)
class Catalogue:
    """Full catalogue. ``popularity`` is ground truth and must not reach recommenders."""

    categories: np.ndarray   # (N,) int
    embeddings: np.ndarray   # (N, d)
    centroids: np.ndarray    # (C, d) unit-norm category centroids
    popularity: np.ndarray   # (N,) pi_i, sums to 1 within each category
    config: CatalogueConfig

    @property
    def n_items(self) -> int:
        return self.categories.shape[0]

    @property
    def n_categories(self) -> int:
        return self.config.n_categories

    def public(self) -> PublicCatalogue:
        return PublicCatalogue(self.categories, self.embeddings, self.n_categories)


def _read_only(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


def generate_catalogue(cfg: CatalogueConfig, rng: np.random.Generator) -> Catalogue:
    """Draw centroids, item embeddings and within-category popularity.

    mu_c  = normalise(h_c),                h_c ~ N(0, I_d)            (independent, D-22)
    phi_i = normalise(mu_{c_i} + sigma*eps_i), eps_i ~ N(0, I_d / d)
    pi_i  proportional to exp(s_pop * zeta_i), zeta_i ~ N(0, 1), normalised per category
    """
    C, n_c, d = cfg.n_categories, cfg.items_per_category, cfg.embedding_dim
    if min(C, n_c, d) < 1:
        raise ValueError("catalogue sizes must be positive")

    centroids = _unit_rows(rng.standard_normal((C, d)))
    categories = np.repeat(np.arange(C), n_c)
    eps = rng.standard_normal((C * n_c, d)) / np.sqrt(d)
    embeddings = _unit_rows(centroids[categories] + cfg.sigma * eps)

    weights = np.exp(cfg.popularity_spread * rng.standard_normal((C, n_c)))
    popularity = (weights / weights.sum(axis=1, keepdims=True)).reshape(-1)

    _read_only(centroids, categories, embeddings, popularity)
    return Catalogue(categories, embeddings, centroids, popularity, cfg)
