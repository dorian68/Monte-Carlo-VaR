from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationResult:
    tickers: list[str]
    drifts: np.ndarray
    vols: np.ndarray
    correlation: np.ndarray


def calibrate_from_returns(returns: pd.DataFrame, trading_days: int = 252) -> CalibrationResult:
    if returns.empty:
        raise ValueError("Returns data is empty.")

    mean_daily = returns.mean()
    vol_daily = returns.std(ddof=1)

    drifts = mean_daily.to_numpy() * trading_days
    vols = vol_daily.to_numpy() * np.sqrt(trading_days)
    correlation = returns.corr().to_numpy()

    return CalibrationResult(
        tickers=list(returns.columns),
        drifts=drifts,
        vols=vols,
        correlation=correlation,
    )
