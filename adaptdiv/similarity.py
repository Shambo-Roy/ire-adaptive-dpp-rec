"""Item similarity kernel (DECISIONS.md D-12).

S = Phi Phi^T + eps I, where the rows of Phi are the public unit-norm item embeddings.
Phi Phi^T is a Gram matrix, so it is positive semidefinite by construction, with rank at
most d. The jitter eps makes S strictly positive definite for numerical stability;
it does not create real diversity, so the embedding dimension must still satisfy d >= K.
"""
from __future__ import annotations

import numpy as np

DEFAULT_JITTER = 1e-6


def similarity_matrix(embeddings: np.ndarray, jitter: float = DEFAULT_JITTER) -> np.ndarray:
    """Cosine similarity of unit-norm embeddings, plus ``jitter`` on the diagonal."""
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    S = embeddings @ embeddings.T
    S[np.diag_indices_from(S)] += jitter
    return S


def min_eigenvalue(S: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix (used to verify positive semidefiniteness)."""
    return float(np.linalg.eigvalsh(S)[0])
