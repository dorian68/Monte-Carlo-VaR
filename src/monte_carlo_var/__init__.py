"""Monte Carlo VaR/ES toolkit."""

from .config import (
    AssetConfig,
    BacktestConfig,
    DataConfig,
    MethodConfig,
    PortfolioConfig,
    ReportConfig,
    RunConfig,
    SimulationConfig,
    StressConfig,
)
from .engine import run_var_methods
from .risk import compute_var_es
from .simulation import SimulationResult, run_monte_carlo

__all__ = [
    "AssetConfig",
    "BacktestConfig",
    "DataConfig",
    "MethodConfig",
    "PortfolioConfig",
    "ReportConfig",
    "RunConfig",
    "SimulationConfig",
    "StressConfig",
    "SimulationResult",
    "compute_var_es",
    "run_monte_carlo",
    "run_var_methods",
]

__version__ = "0.3.0"
