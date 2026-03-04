from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .calibration import calibrate_from_returns
from .config import MethodConfig, RunConfig
from .market_data import compute_returns, load_price_data
from .portfolio import Portfolio
from .risk import (
    VarResult,
    cornish_fisher_var_es,
    evt_var_es,
    filtered_historical_var_es,
    historical_var_es,
    parametric_normal_var_es,
    student_t_var_es,
)
from .risk.var import compute_var_es
from .simulation import simulate_portfolio_pnl


HISTORICAL_METHODS = {
    "historical",
    "parametric_normal",
    "cornish_fisher",
    "filtered_historical",
    "evt",
    "student_t",
}


def run_var_methods(cfg: RunConfig, methods_override: Iterable[str] | None = None) -> list[VarResult]:
    portfolio = Portfolio(cfg.portfolio.assets)
    tickers = portfolio.tickers()

    methods = _select_methods(cfg, methods_override)

    returns = None
    if _needs_returns(methods, cfg):
        returns = _load_returns(cfg, tickers)

    results: list[VarResult] = []
    for method in methods:
        results.append(_run_method(cfg, portfolio, method, returns))

    return results


def _select_methods(cfg: RunConfig, methods_override: Iterable[str] | None) -> list[MethodConfig]:
    if methods_override:
        override = {name.strip() for name in methods_override if name.strip()}
        selected = [method for method in cfg.methods if method.name in override]
        known = {method.name for method in selected}
        for name in override - known:
            selected.append(MethodConfig(name=name))
        return selected
    return cfg.methods


def _needs_returns(methods: list[MethodConfig], cfg: RunConfig) -> bool:
    if cfg.backtest or cfg.stress:
        return True
    return any(method.name in HISTORICAL_METHODS for method in methods)


def _load_returns(cfg: RunConfig, tickers: list[str]) -> pd.DataFrame:
    if not cfg.data.prices_path:
        raise ValueError("data.prices_path is required for historical methods/backtests/stress tests.")

    prices = load_price_data(cfg.data.prices_path, cfg.data.date_column)
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Price data missing tickers: {', '.join(missing)}")

    returns = compute_returns(prices[tickers], cfg.data.return_type)
    if returns.empty:
        raise ValueError("Returns data is empty after computation.")
    return returns


def _run_method(
    cfg: RunConfig,
    portfolio: Portfolio,
    method: MethodConfig,
    returns: pd.DataFrame | None,
) -> VarResult:
    name = method.name
    params = method.params
    confidence = float(params.get("confidence", cfg.simulation.confidence))

    if name == "monte_carlo":
        return _monte_carlo_var(cfg, portfolio, params, returns, confidence)

    if returns is None:
        raise ValueError(f"Method '{name}' requires historical returns.")

    returns = _apply_lookback(returns, params)

    weights = portfolio.weights()
    portfolio_value = portfolio.value()

    if name == "historical":
        return historical_var_es(returns, weights, portfolio_value, confidence)
    if name == "parametric_normal":
        return parametric_normal_var_es(returns, weights, portfolio_value, confidence)
    if name == "cornish_fisher":
        return cornish_fisher_var_es(returns, weights, portfolio_value, confidence)
    if name == "filtered_historical":
        lambda_ = float(params.get("lambda", 0.94))
        return filtered_historical_var_es(returns, weights, portfolio_value, confidence, lambda_)
    if name == "evt":
        threshold = float(params.get("threshold", 0.95))
        return evt_var_es(returns, weights, portfolio_value, confidence, threshold)
    if name == "student_t":
        df = int(params.get("df", 6))
        return student_t_var_es(returns, weights, portfolio_value, confidence, df)

    raise ValueError(f"Unknown method: {name}")


def _monte_carlo_var(
    cfg: RunConfig,
    portfolio: Portfolio,
    params: dict,
    returns: pd.DataFrame | None,
    confidence: float,
) -> VarResult:
    simulations = int(params.get("simulations", cfg.simulation.simulations))
    horizon_days = int(params.get("horizon_days", cfg.simulation.horizon_days))
    trading_days = int(params.get("trading_days", cfg.simulation.trading_days))
    seed = params.get("seed", cfg.simulation.seed)

    if returns is not None and bool(params.get("calibrate", False)):
        calibration = calibrate_from_returns(returns, trading_days=trading_days)
        drifts = calibration.drifts
        vols = calibration.vols
        correlation = calibration.correlation
        calibrated = True
    else:
        drifts = portfolio.drifts()
        vols = portfolio.vols()
        correlation = np.array(cfg.portfolio.correlation, dtype=float)
        calibrated = False

    pnl = simulate_portfolio_pnl(
        prices0=portfolio.prices(),
        quantities=portfolio.quantities(),
        drifts=drifts,
        vols=vols,
        correlation=correlation,
        horizon_days=horizon_days,
        simulations=simulations,
        trading_days=trading_days,
        seed=seed,
    )

    var, es = compute_var_es(pnl, confidence)
    return VarResult(
        method="monte_carlo",
        var=var,
        es=es,
        confidence=confidence,
        pnl=pnl,
        metadata={\n+            \"simulations\": simulations,\n+            \"horizon_days\": horizon_days,\n+            \"calibrated\": calibrated,\n+        },\n+    )


def _apply_lookback(returns: pd.DataFrame, params: dict) -> pd.DataFrame:
    lookback = params.get("lookback")
    if lookback is None:
        return returns
    lookback = int(lookback)
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    return returns.tail(lookback)
