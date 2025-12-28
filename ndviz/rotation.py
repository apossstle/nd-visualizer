from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


def rotate_plane_inplace(X: np.ndarray, i: int, j: int, theta: float) -> None:
    if i == j:
        raise ValueError("i and j must be different axes")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (m, n)")
    m, n = X.shape
    if not (0 <= i < n and 0 <= j < n):
        raise IndexError("axis indices out of range")
    c = np.cos(theta)
    s = np.sin(theta)
    xi = X[:, i].copy()
    xj = X[:, j].copy()
    X[:, i] = c * xi - s * xj
    X[:, j] = s * xi + c * xj


@dataclass
class NDSpin:
    pairs: List[Tuple[int, int]]
    omegas: np.ndarray
    phases: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.phases is not None and len(self.phases) != len(self.pairs):
            raise ValueError("length of phases must match length of pairs")
        if len(self.omegas) != len(self.pairs):
            raise ValueError("length of omegas must match length of pairs")

    def apply(self, X: np.ndarray, t: float) -> np.ndarray:
        Y = X.copy()
        for idx, (i, j) in enumerate(self.pairs):
            omega = self.omegas[idx]
            phase = 0.0 if self.phases is None else self.phases[idx]
            theta = omega * t + phase
            rotate_plane_inplace(Y, i, j, theta)
        return Y


def default_spin(n: int, seed: Optional[int] = None) -> NDSpin:
    if n < 2:
        raise ValueError("n must be >= 2 to define at least one rotation plane")
    pairs: List[Tuple[int, int]] = []
    for i in range(0, n - 1, 2):
        pairs.append((i, i + 1))
    rng = np.random.default_rng(seed)
    omegas = rng.uniform(0.5, 1.5, size=len(pairs))
    phases = rng.uniform(0.0, 2 * np.pi, size=len(pairs))
    return NDSpin(pairs=pairs, omegas=omegas, phases=phases)