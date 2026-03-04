from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AssetConfig:
    ticker: str
    price: float
    volatility: float
    quantity: float
    drift: float = 0.0


@dataclass(frozen=True)
class PortfolioConfig:
    assets: list[AssetConfig]
    correlation: list[list[float]]


@dataclass(frozen=True)
class SimulationConfig:
    confidence: float
    horizon_days: int
    simulations: int
    seed: int | None = None
    trading_days: int = 252


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "outputs"
    prefix: str = "run"


@dataclass(frozen=True)
class DataConfig:
    prices_path: str | None = None
    date_column: str = "date"
    return_type: str = "log"


@dataclass(frozen=True)
class MethodConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestConfig:
    method: str
    window: int
    confidence: float
    step: int = 1


@dataclass(frozen=True)
class StressScenarioConfig:
    name: str
    returns: dict[str, float]


@dataclass(frozen=True)
class StressConfig:
    scenarios: list[StressScenarioConfig] = field(default_factory=list)
    historical_top_n: int | None = None


@dataclass(frozen=True)
class ReportConfig:
    html: bool = True


@dataclass(frozen=True)
class RunConfig:
    portfolio: PortfolioConfig
    simulation: SimulationConfig
    output: OutputConfig = OutputConfig()
    data: DataConfig = DataConfig()
    methods: list[MethodConfig] = field(default_factory=list)
    backtest: BacktestConfig | None = None
    stress: StressConfig | None = None
    report: ReportConfig = ReportConfig()


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    portfolio_raw = _require(raw, "portfolio")
    assets_raw = _require(portfolio_raw, "assets")
    correlation = _require(portfolio_raw, "correlation")

    assets = [
        AssetConfig(
            ticker=str(_require(asset, "ticker")),
            price=float(_require(asset, "price")),
            volatility=float(_require(asset, "volatility")),
            quantity=float(_require(asset, "quantity")),
            drift=float(asset.get("drift", 0.0)),
        )
        for asset in assets_raw
    ]

    simulation_raw = _require(raw, "simulation")
    simulation = SimulationConfig(
        confidence=float(_require(simulation_raw, "confidence")),
        horizon_days=int(_require(simulation_raw, "horizon_days")),
        simulations=int(_require(simulation_raw, "simulations")),
        seed=(int(simulation_raw["seed"]) if "seed" in simulation_raw else None),
        trading_days=int(simulation_raw.get("trading_days", 252)),
    )

    output_raw = raw.get("output", {})
    output = OutputConfig(
        directory=str(output_raw.get("directory", "outputs")),
        prefix=str(output_raw.get("prefix", "run")),
    )

    data_raw = raw.get("data", {})
    data = DataConfig(
        prices_path=str(data_raw["prices_path"]) if "prices_path" in data_raw else None,
        date_column=str(data_raw.get("date_column", "date")),
        return_type=str(data_raw.get("return_type", "log")),
    )

    methods = _parse_methods(raw)

    backtest_raw = raw.get("backtest")
    backtest = None
    if backtest_raw:
        backtest = BacktestConfig(
            method=str(_require(backtest_raw, "method")),
            window=int(_require(backtest_raw, "window")),
            confidence=float(_require(backtest_raw, "confidence")),
            step=int(backtest_raw.get("step", 1)),
        )

    stress_raw = raw.get("stress")
    stress = None
    if stress_raw:
        scenarios = [
            StressScenarioConfig(
                name=str(_require(item, "name")),
                returns={str(k): float(v) for k, v in _require(item, "returns").items()},
            )
            for item in stress_raw.get("scenarios", [])
        ]
        historical_raw = stress_raw.get("historical")
        historical_top_n = None
        if historical_raw and "top_n" in historical_raw:
            historical_top_n = int(historical_raw["top_n"])
        stress = StressConfig(scenarios=scenarios, historical_top_n=historical_top_n)

    report_raw = raw.get("report", {})
    report = ReportConfig(html=bool(report_raw.get("html", True)))

    run_config = RunConfig(
        portfolio=PortfolioConfig(assets=assets, correlation=correlation),
        simulation=simulation,
        output=output,
        data=data,
        methods=methods,
        backtest=backtest,
        stress=stress,
        report=report,
    )
    validate_config(run_config)
    return run_config


def validate_config(cfg: RunConfig) -> None:
    assets = cfg.portfolio.assets
    if not assets:
        raise ValueError("Portfolio must contain at least one asset.")

    for asset in assets:
        if asset.price <= 0:
            raise ValueError(f"Asset {asset.ticker} has non-positive price.")
        if asset.volatility <= 0:
            raise ValueError(f"Asset {asset.ticker} has non-positive volatility.")

    confidence = cfg.simulation.confidence
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be in (0, 1).")

    if cfg.simulation.horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if cfg.simulation.simulations <= 0:
        raise ValueError("simulations must be positive.")
    if cfg.simulation.trading_days <= 0:
        raise ValueError("trading_days must be positive.")

    if cfg.data.return_type not in {"log", "simple"}:
        raise ValueError("data.return_type must be 'log' or 'simple'.")

    correlation = cfg.portfolio.correlation
    n_assets = len(assets)
    if len(correlation) != n_assets:
        raise ValueError("Correlation matrix must be square and match asset count.")

    for row in correlation:
        if len(row) != n_assets:
            raise ValueError("Correlation matrix must be square and match asset count.")
        for value in row:
            if value < -1.0 or value > 1.0:
                raise ValueError("Correlation values must be between -1 and 1.")

    for idx in range(n_assets):
        if abs(correlation[idx][idx] - 1.0) > 1e-6:
            raise ValueError("Correlation diagonal must be 1.")

    if cfg.backtest:
        if cfg.backtest.window <= 0:
            raise ValueError("backtest.window must be positive.")
        if cfg.backtest.step <= 0:
            raise ValueError("backtest.step must be positive.")
        if not (0.0 < cfg.backtest.confidence < 1.0):
            raise ValueError("backtest.confidence must be in (0, 1).")


def _parse_methods(raw: dict) -> list[MethodConfig]:
    methods_raw = raw.get("methods")
    if methods_raw is None and "method" in raw:
        methods_raw = [raw["method"]]

    if not methods_raw:
        return [MethodConfig(name="monte_carlo")]

    methods: list[MethodConfig] = []
    for item in methods_raw:
        if isinstance(item, str):
            methods.append(MethodConfig(name=item))
            continue
        if isinstance(item, dict):
            name = str(_require(item, "name"))
            params = dict(item.get("params", {}))
            methods.append(MethodConfig(name=name, params=params))
            continue
        raise ValueError("method entries must be string or object with name/params.")
    return methods


def _require(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(f"Missing required field: {key}")
    return mapping[key]
