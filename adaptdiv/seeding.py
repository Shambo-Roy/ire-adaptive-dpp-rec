"""Reproducible random streams (SIMULATION_DESIGN.md section 7).

Every random quantity is drawn from a stream addressed by a fixed *path* of integers
under the run seed, e.g. ``(USERS, kappa_index, user_index, INTERACTIONS)``. Streams are
built with ``SeedSequence(entropy, spawn_key=path)``, so a stream depends only on
(seed, path): adding users, scenarios or kappa levels never changes existing draws.
"""
from __future__ import annotations

import numpy as np

# Fixed tag mixed into every seed so streams are specific to this project.
PROJECT_TAG = 20261026

# Top-level stream indices.
CATALOGUE = 0
TRAINING = 1
USERS = 2

# Per-user stream indices.
PARAMS = 0        # z_A, half-swap permutation, taste u
INTERACTIONS = 1  # pre-drawn uniform random-number bank


def stream(seed: int, *path: int) -> np.random.Generator:
    """Return the generator for ``path`` under run ``seed``."""
    if seed < 0 or any(p < 0 for p in path):
        raise ValueError("seed and path entries must be non-negative")
    ss = np.random.SeedSequence(entropy=[PROJECT_TAG, seed], spawn_key=tuple(path))
    return np.random.default_rng(ss)
