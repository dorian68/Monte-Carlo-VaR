from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import StressConfig
from .portfolio import Portfolio
from .risk.methods import portfolio_returns


@dataclass(frozen=True)
class StressResult:
    name: str
    portfolio_return: float
    pnl: float


def run_stress_tests(
    portfolio: Portfolio,
    stress: StressConfig,
    historical_returns: pd.DataFrame | None = None,
) -> list[StressResult]:
    results: list[StressResult] = []

    tickers = portfolio.tickers()
    weights = portfolio.weights()
    portfolio_value = portfolio.value()

    for scenario in stress.scenarios:
        scenario_returns = np.array([scenario.returns.get(ticker, 0.0) for ticker in tickers], dtype=float)
        port_return = float(np.dot(weights, scenario_returns))
        pnl = port_return * portfolio_value
        results.append(StressResult(name=scenario.name, portfolio_return=port_return, pnl=pnl))

    if stress.historical_top_n and historical_returns is not None and not historical_returns.empty:
        port_returns = portfolio_returns(historical_returns, weights)
        series = pd.Series(port_returns, index=historical_returns.index)
        worst = series.nsmallest(stress.historical_top_n)
        for idx, value in worst.items():
            name = f"Historical {idx.date()}" if hasattr(idx, "date") else f"Historical {idx}"
            pnl = float(value * portfolio_value)
            results.append(StressResult(name=name, portfolio_return=float(value), pnl=pnl))

    return results
