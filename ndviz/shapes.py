from __future__ import annotations

import itertools
import numpy as np


def hypercube_pm1(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    return np.array(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=float)


def hypersphere(n: int, num_points: int, seed: int | None = None) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    if num_points < 1:
        raise ValueError("num_points must be >= 1")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num_points, n))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


