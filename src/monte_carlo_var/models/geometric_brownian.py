from __future__ import annotations

import numpy as np


def simulate_log_returns(
    mean: np.ndarray,
    covariance: np.ndarray,
    simulations: int,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, covariance, size=simulations)
