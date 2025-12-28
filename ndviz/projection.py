from __future__ import annotations

import numpy as np


def random_orthonormal_basis(n: int, out_dim: int = 3, seed: int | None = None) -> np.ndarray:
    if out_dim < 1:
        raise ValueError("out_dim must be >= 1")
    if n < out_dim:
        raise ValueError("n must be >= out_dim for projection basis")
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, out_dim))
    Q, _ = np.linalg.qr(A)
    P = Q.T 
    return P


def project(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (k, n)")
    if P.ndim != 2:
        raise ValueError("P must be a 2D array of shape (m, n)")
    if X.shape[1] != P.shape[1]:
        raise ValueError("dimension mismatch between X and projection basis P")
    return X @ P.T