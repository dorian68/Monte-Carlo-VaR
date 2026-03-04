import numpy as np
import pandas as pd

from monte_carlo_var.risk import (
    historical_var_es,
    parametric_normal_var_es,
    cornish_fisher_var_es,
)


def test_method_outputs_are_finite():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.015, -0.005, 0.007],
            "B": [0.008, -0.01, 0.012, -0.004, 0.006],
        }
    )
    weights = np.array([0.6, 0.4])
    portfolio_value = 100.0

    for func in (historical_var_es, parametric_normal_var_es, cornish_fisher_var_es):
        result = func(returns, weights, portfolio_value, 0.8)
        assert np.isfinite(result.var)
        assert np.isfinite(result.es)
