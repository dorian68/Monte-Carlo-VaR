from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import RunConfig
from .models.geometric_brownian import simulate_log_returns
from .portfolio import Portfolio
from .risk import compute_var_es


@dataclass(frozen=True)
class SimulationResult:
    pnl: np.ndarray
    var: float
    es: float
    mean: float
    stdev: float
    portfolio_value: float
    confidence: float
    horizon_days: int
    simulations: int


def run_monte_carlo(cfg: RunConfig) -> SimulationResult:
    portfolio = Portfolio(cfg.portfolio.assets)
    prices0 = portfolio.prices()
    quantities = portfolio.quantities()
    vols = portfolio.vols()
    drifts = portfolio.drifts()

    correlation = np.array(cfg.portfolio.correlation, dtype=float)
    portfolio_pnl = simulate_portfolio_pnl(
        prices0=prices0,
        quantities=quantities,
        drifts=drifts,
        vols=vols,
        correlation=correlation,
        horizon_days=cfg.simulation.horizon_days,
        simulations=cfg.simulation.simulations,
        trading_days=cfg.simulation.trading_days,
        seed=cfg.simulation.seed,
    )

    var, es = compute_var_es(portfolio_pnl, cfg.simulation.confidence)

    return SimulationResult(
        pnl=portfolio_pnl,
        var=var,
        es=es,
        mean=float(np.mean(portfolio_pnl)),
        stdev=float(np.std(portfolio_pnl, ddof=1)),
        portfolio_value=portfolio.value(),
        confidence=cfg.simulation.confidence,
        horizon_days=cfg.simulation.horizon_days,
        simulations=cfg.simulation.simulations,
    )


def _covariance_from_vols(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if corr.shape[0] != vols.shape[0]:
        raise ValueError("Correlation matrix size must match asset count.")
    return np.outer(vols, vols) * corr


def simulate_portfolio_pnl(
    prices0: np.ndarray,
    quantities: np.ndarray,
    drifts: np.ndarray,
    vols: np.ndarray,
    correlation: np.ndarray,
    horizon_days: int,
    simulations: int,
    trading_days: int,
    seed: int | None = None,
) -> np.ndarray:
    covariance = _covariance_from_vols(vols, correlation)

    scale = horizon_days / trading_days
    mean = (drifts - 0.5 * vols**2) * scale
    covariance = covariance * scale

    log_returns = simulate_log_returns(
        mean,
        covariance,
        simulations=simulations,
        seed=seed,
    )

    prices_t = prices0 * np.exp(log_returns)
    pnl = (prices_t - prices0) * quantities
    return np.sum(pnl, axis=1)
