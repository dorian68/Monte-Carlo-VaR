from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .calibration import calibrate_from_returns
from .engine import _apply_lookback
from .portfolio import Portfolio
from .risk import VarResult
from .risk.var import compute_var_es
from .risk.methods import portfolio_returns
from .config import MethodConfig
from .simulation import simulate_portfolio_pnl


@dataclass(frozen=True)
class BacktestResult:
    exceptions: int
    observations: int
    exception_rate: float
    lr_uc: float
    p_value_uc: float
    lr_ind: float
    p_value_ind: float
    lr_cc: float
    p_value_cc: float


@dataclass(frozen=True)
class RollingBacktestOutput:
    var_series: np.ndarray
    pnl_series: np.ndarray
    results: BacktestResult


def backtest_var(pnl: np.ndarray, var: float | np.ndarray, confidence: float) -> BacktestResult:
    pnl = np.asarray(pnl, dtype=float)
    losses = -pnl

    var_array = np.asarray(var, dtype=float)
    if var_array.ndim == 0:
        var_array = np.full_like(losses, float(var_array))

    if var_array.shape != losses.shape:
        raise ValueError("VaR series must match PnL series length.")

    exceptions = losses > var_array
    n = len(losses)
    x = int(np.sum(exceptions))
    exception_rate = x / n if n else 0.0

    lr_uc, p_uc = _kupiec_lr(x, n, confidence)
    lr_ind, p_ind = _christoffersen_lr(exceptions)
    lr_cc = lr_uc + lr_ind
    p_cc = float(1.0 - chi2.cdf(lr_cc, df=2))

    return BacktestResult(
        exceptions=x,
        observations=n,
        exception_rate=exception_rate,
        lr_uc=lr_uc,
        p_value_uc=p_uc,
        lr_ind=lr_ind,
        p_value_ind=p_ind,
        lr_cc=lr_cc,
        p_value_cc=p_cc,
    )


def rolling_backtest(
    returns: pd.DataFrame,
    portfolio: Portfolio,
    method: MethodConfig,
    confidence: float,
    window: int,
    step: int = 1,
) -> RollingBacktestOutput:
    if window <= 1:
        raise ValueError("window must be > 1.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if len(returns) <= window:
        raise ValueError("Not enough returns for the requested backtest window.")

    weights = portfolio.weights()
    portfolio_value = portfolio.value()

    var_series: list[float] = []
    pnl_series: list[float] = []

    for idx in range(window, len(returns), step):
        window_slice = returns.iloc[idx - window : idx]
        window_slice = _apply_lookback(window_slice, method.params)
        var_result = _run_method_for_backtest(portfolio, method, window_slice, confidence)

        realized_return = float(portfolio_returns(returns.iloc[idx : idx + 1], weights)[0])
        pnl_series.append(realized_return * portfolio_value)
        var_series.append(var_result.var)

    pnl_array = np.array(pnl_series, dtype=float)
    var_array = np.array(var_series, dtype=float)
    results = backtest_var(pnl_array, var_array, confidence)

    return RollingBacktestOutput(var_series=var_array, pnl_series=pnl_array, results=results)


def _run_method_for_backtest(
    portfolio: Portfolio,
    method: MethodConfig,
    returns: pd.DataFrame,
    confidence: float,
) -> VarResult:
    if method.name == "monte_carlo":
        simulations = int(method.params.get("simulations", 2000))
        horizon_days = int(method.params.get("horizon_days", 1))
        trading_days = int(method.params.get("trading_days", 252))
        seed = method.params.get("seed")

        calibration = calibrate_from_returns(returns, trading_days=trading_days)
        pnl = simulate_portfolio_pnl(
            prices0=portfolio.prices(),
            quantities=portfolio.quantities(),
            drifts=calibration.drifts,
            vols=calibration.vols,
            correlation=calibration.correlation,
            horizon_days=horizon_days,
            simulations=simulations,
            trading_days=trading_days,
            seed=seed,
        )
        var, es = compute_var_es(pnl, confidence)
        return VarResult(method="monte_carlo", var=var, es=es, confidence=confidence)

    weights = portfolio.weights()
    portfolio_value = portfolio.value()
    params = method.params

    if method.name == "historical":
        return _run_method_simple("historical", returns, weights, portfolio_value, confidence, params)
    if method.name == "parametric_normal":
        return _run_method_simple("parametric_normal", returns, weights, portfolio_value, confidence, params)
    if method.name == "cornish_fisher":
        return _run_method_simple("cornish_fisher", returns, weights, portfolio_value, confidence, params)
    if method.name == "filtered_historical":
        return _run_method_simple("filtered_historical", returns, weights, portfolio_value, confidence, params)
    if method.name == "evt":
        return _run_method_simple("evt", returns, weights, portfolio_value, confidence, params)
    if method.name == "student_t":
        return _run_method_simple("student_t", returns, weights, portfolio_value, confidence, params)

    raise ValueError(f"Unsupported method for backtest: {method.name}")


def _run_method_simple(
    name: str,
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
    params: dict,
) -> VarResult:
    from .risk import (
        cornish_fisher_var_es,
        evt_var_es,
        filtered_historical_var_es,
        historical_var_es,
        parametric_normal_var_es,
        student_t_var_es,
    )

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


def _kupiec_lr(exceptions: int, observations: int, confidence: float) -> tuple[float, float]:
    if observations == 0:
        return 0.0, 1.0

    p = 1.0 - confidence
    epsilon = 1e-12
    p = min(max(p, epsilon), 1.0 - epsilon)

    phat = exceptions / observations
    phat = min(max(phat, epsilon), 1.0 - epsilon)

    log_likelihood_null = (observations - exceptions) * np.log(1.0 - p) + exceptions * np.log(p)
    log_likelihood_alt = (observations - exceptions) * np.log(1.0 - phat) + exceptions * np.log(phat)

    lr_stat = -2.0 * (log_likelihood_null - log_likelihood_alt)
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value


def _christoffersen_lr(exceptions: np.ndarray) -> tuple[float, float]:
    if len(exceptions) < 2:
        return 0.0, 1.0

    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(exceptions[:-1], exceptions[1:]):
        if not prev and not curr:
            n00 += 1
        elif not prev and curr:
            n01 += 1
        elif prev and not curr:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11

    p01 = n01 / n0 if n0 > 0 else 0.0
    p11 = n11 / n1 if n1 > 0 else 0.0
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    epsilon = 1e-12
    p01 = min(max(p01, epsilon), 1.0 - epsilon)
    p11 = min(max(p11, epsilon), 1.0 - epsilon)
    p = min(max(p, epsilon), 1.0 - epsilon)

    log_likelihood_ind = (
        n00 * np.log(1.0 - p01)
        + n01 * np.log(p01)
        + n10 * np.log(1.0 - p11)
        + n11 * np.log(p11)
    )
    log_likelihood_null = (
        (n00 + n10) * np.log(1.0 - p) + (n01 + n11) * np.log(p)
    )

    lr_stat = -2.0 * (log_likelihood_null - log_likelihood_ind)
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value
