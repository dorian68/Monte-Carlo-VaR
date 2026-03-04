import numpy as np

from monte_carlo_var.backtest import backtest_var


def test_backtest_var_counts_exceptions():
    pnl = np.array([-5, 1, -2, -7, 3], dtype=float)
    result = backtest_var(pnl, var=4.0, confidence=0.99)

    assert result.exceptions == 2
    assert result.observations == 5
    assert result.exception_rate == 2 / 5
