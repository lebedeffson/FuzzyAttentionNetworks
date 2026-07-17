from __future__ import annotations

import numpy as np


def sample_random_directions(n: int, dim: int, norm: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x * float(norm)
