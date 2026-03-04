from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import rolling_backtest
from .config import MethodConfig, load_config
from .engine import run_var_methods
from .market_data import compute_returns, load_price_data
from .portfolio import Portfolio
from .reporting import (
    save_backtest_json,
    save_html_report,
    save_methods_json,
    save_pnl_csv,
    save_pnl_histogram,
    save_results_json,
    save_stress_csv,
)
from .stress import run_stress_tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo VaR/ES engine")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--out", default=None, help="Output directory override")
    parser.add_argument("--prefix", default=None, help="Output filename prefix override")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--methods", default=None, help="Comma-separated method names override")
    parser.add_argument("--no-report", action="store_true", help="Skip plots + HTML report")

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.out:
        cfg = replace(cfg, output=replace(cfg.output, directory=args.out))
    if args.prefix:
        cfg = replace(cfg, output=replace(cfg.output, prefix=args.prefix))
    if args.seed is not None:
        cfg = replace(cfg, simulation=replace(cfg.simulation, seed=args.seed))

    methods_override = None
    if args.methods:
        methods_override = [name.strip() for name in args.methods.split(",") if name.strip()]

    results = run_var_methods(cfg, methods_override=methods_override)

    out_dir = Path(cfg.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.output.prefix

    plots: dict[str, str] = {}

    for result in results:
        method_prefix = f"{prefix}_{result.method}"
        results_json = out_dir / f"{method_prefix}_results.json"
        save_results_json(result, results_json)

        if result.pnl is not None:
            pnl_csv = out_dir / f"{method_prefix}_pnl.csv"
            save_pnl_csv(result.pnl, pnl_csv)

            if not args.no_report:
                plot_path = out_dir / f"{method_prefix}_pnl.png"
                save_pnl_histogram(result.pnl, result.var, result.es, plot_path, title=f"{result.method} PnL")
                plots[result.method] = plot_path.name

    methods_json = out_dir / f"{prefix}_methods.json"
    save_methods_json(results, methods_json)

    portfolio = Portfolio(cfg.portfolio.assets)

    returns = None
    if (cfg.backtest or cfg.stress) and cfg.data.prices_path:
        prices = load_price_data(cfg.data.prices_path, cfg.data.date_column)
        prices = prices[portfolio.tickers()]
        returns = compute_returns(prices, cfg.data.return_type)
    if (cfg.backtest or cfg.stress) and returns is None:
        raise ValueError(\"data.prices_path is required for backtest/stress.\")

    backtest_result = None
    if cfg.backtest and returns is not None:
        method_cfg = _resolve_method(cfg, cfg.backtest.method)
        backtest_output = rolling_backtest(
            returns=returns,
            portfolio=portfolio,
            method=method_cfg,
            confidence=cfg.backtest.confidence,
            window=cfg.backtest.window,
            step=cfg.backtest.step,
        )
        backtest_result = backtest_output.results
        save_backtest_json(backtest_result, out_dir / f"{prefix}_backtest.json")

        backtest_series = pd.DataFrame(
            {"pnl": backtest_output.pnl_series, "var": backtest_output.var_series}
        )
        backtest_series.to_csv(out_dir / f"{prefix}_backtest_series.csv", index=False)

    stress_results = None
    if cfg.stress and returns is not None:
        stress_results = run_stress_tests(portfolio, cfg.stress, returns)
        save_stress_csv(stress_results, out_dir / f"{prefix}_stress.csv")

    if not args.no_report and cfg.report.html:
        save_html_report(
            results=results,
            plots=plots,
            backtest=backtest_result,
            stress=stress_results,
            path=out_dir / f"{prefix}_report.html",
        )

    print("VaR run completed")
    for result in results:
        print(f"{result.method}: VaR={result.var:,.2f} | ES={result.es:,.2f}")
    print(f"Outputs: {out_dir.resolve()}")


def _resolve_method(cfg, name: str) -> MethodConfig:
    for method in cfg.methods:
        if method.name == name:
            return method
    return MethodConfig(name=name)


if __name__ == "__main__":
    main()
