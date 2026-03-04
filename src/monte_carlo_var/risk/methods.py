from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t, genpareto, skew, kurtosis

from .var import compute_var_es


@dataclass(frozen=True)
class VarResult:
    method: str
    var: float
    es: float
    confidence: float
    pnl: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    return returns.to_numpy() @ weights


def historical_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
) -> VarResult:
    port_returns = portfolio_returns(returns, weights)
    pnl = port_returns * portfolio_value
    var, es = compute_var_es(pnl, confidence)
    return VarResult(method="historical", var=var, es=es, confidence=confidence, pnl=pnl)


def parametric_normal_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
) -> VarResult:
    port_returns = portfolio_returns(returns, weights)
    mu = float(np.mean(port_returns))
    sigma = float(np.std(port_returns, ddof=1))

    mean_loss = -portfolio_value * mu
    scale_loss = portfolio_value * sigma
    if sigma == 0.0:
        var = mean_loss
        es = mean_loss
    else:
        z = norm.ppf(confidence)
        var = mean_loss + scale_loss * z
        es = mean_loss + scale_loss * (norm.pdf(z) / (1.0 - confidence))

    return VarResult(
        method="parametric_normal",
        var=float(var),
        es=float(es),
        confidence=confidence,
        metadata={"mu": mu, "sigma": sigma},
    )


def cornish_fisher_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
) -> VarResult:
    port_returns = portfolio_returns(returns, weights)
    mu = float(np.mean(port_returns))
    sigma = float(np.std(port_returns, ddof=1))

    skewness = float(skew(port_returns, bias=False))
    excess_kurtosis = float(kurtosis(port_returns, fisher=True, bias=False))

    mean_loss = -portfolio_value * mu
    scale_loss = portfolio_value * sigma
    if sigma == 0.0:
        var = mean_loss
        es = mean_loss
        z_cf = 0.0
    else:
        z = norm.ppf(confidence)
        z_cf = (
            z
            + (1.0 / 6.0) * (z**2 - 1.0) * skewness
            + (1.0 / 24.0) * (z**3 - 3.0 * z) * excess_kurtosis
            - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * (skewness**2)
        )
        var = mean_loss + scale_loss * z_cf
        es = mean_loss + scale_loss * (norm.pdf(z_cf) / (1.0 - confidence))

    return VarResult(
        method="cornish_fisher",
        var=float(var),
        es=float(es),
        confidence=confidence,
        metadata={
            "mu": mu,
            "sigma": sigma,
            "skew": skewness,
            "excess_kurtosis": excess_kurtosis,
            "z_cf": z_cf,
        },
    )


def filtered_historical_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
    lambda_: float = 0.94,
) -> VarResult:
    port_returns = portfolio_returns(returns, weights)
    losses = -port_returns

    sigma2 = np.zeros_like(losses)
    initial = max(5, min(20, len(losses)))
    sigma2[0] = float(np.var(losses[:initial], ddof=1)) if initial > 1 else float(np.var(losses))

    for idx in range(1, len(losses)):
        sigma2[idx] = lambda_ * sigma2[idx - 1] + (1.0 - lambda_) * losses[idx - 1] ** 2

    sigma = np.sqrt(sigma2)
    standardized = losses / sigma

    q = float(np.quantile(standardized, confidence))
    tail = standardized[standardized >= q]
    es_std = float(np.mean(tail)) if len(tail) > 0 else q
    current_sigma = sigma[-1]
    var = q * current_sigma * portfolio_value
    es = es_std * current_sigma * portfolio_value

    pnl = -losses * portfolio_value

    return VarResult(
        method="filtered_historical",
        var=float(var),
        es=float(es),
        confidence=confidence,
        pnl=pnl,
        metadata={"lambda": lambda_},
    )


def student_t_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
    df: int = 6,
) -> VarResult:
    if df <= 2:
        raise ValueError("Student-t df must be > 2 for finite variance.")

    port_returns = portfolio_returns(returns, weights)
    mu = float(np.mean(port_returns))
    sigma = float(np.std(port_returns, ddof=1))

    mean_loss = -portfolio_value * mu
    if sigma == 0.0:
        var = mean_loss
        es = mean_loss
    else:
        scale = sigma * np.sqrt((df - 2) / df)
        z = student_t.ppf(confidence, df)

        scale_loss = portfolio_value * scale
        var = mean_loss + scale_loss * z

        pdf_z = student_t.pdf(z, df)
        es = mean_loss + scale_loss * ((df + z**2) / (df - 1.0)) * (pdf_z / (1.0 - confidence))

    return VarResult(
        method="student_t",
        var=float(var),
        es=float(es),
        confidence=confidence,
        metadata={"mu": mu, "sigma": sigma, "df": df},
    )


def evt_var_es(
    returns: pd.DataFrame,
    weights: np.ndarray,
    portfolio_value: float,
    confidence: float,
    threshold: float = 0.95,
) -> VarResult:
    port_returns = portfolio_returns(returns, weights)
    losses = -port_returns

    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0, 1).")
    if confidence <= threshold:
        raise ValueError("confidence must be greater than threshold for EVT.")

    u = float(np.quantile(losses, threshold))
    exceedances = losses[losses > u] - u

    if len(exceedances) < 10:
        raise ValueError("Not enough exceedances for EVT fit.")

    shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    tail_prob = len(exceedances) / len(losses)

    if shape != 0:
        var_loss = u + (scale / shape) * (((tail_prob / (1.0 - confidence)) ** shape) - 1.0)
    else:
        var_loss = u + scale * np.log(tail_prob / (1.0 - confidence))

    if shape < 1.0:
        es_loss = var_loss + (scale + shape * (var_loss - u)) / (1.0 - shape)
    else:
        es_loss = float("nan")

    return VarResult(
        method="evt",
        var=float(var_loss * portfolio_value),
        es=float(es_loss * portfolio_value),
        confidence=confidence,
        metadata={"threshold": threshold, "shape": shape, "scale": scale},
    )
