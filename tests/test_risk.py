import numpy as np

from monte_carlo_var.risk import compute_var_es


def test_var_es_higher_quantile():
    pnl = np.array([-10, -5, -1, 0, 1, 5, 10], dtype=float)
    var, es = compute_var_es(pnl, 0.8)
    assert var == 5.0
    assert es == 7.5
