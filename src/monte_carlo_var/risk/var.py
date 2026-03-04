from __future__ import annotations

import numpy as np


def compute_var_es(pnl: np.ndarray, confidence: float) -> tuple[float, float]:
    if pnl.ndim != 1:
        raise ValueError("PnL array must be one-dimensional.")

    losses = -pnl
    var = float(np.quantile(losses, confidence, method="higher"))
    tail_losses = losses[losses >= var]
    es = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var
    return var, es
