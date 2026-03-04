# Monte-Carlo-VaR (Institutional Demo)

A production-style, multi-method VaR/ES toolkit with calibration, rolling backtests, stress testing, and HTML reporting. This repo is designed as a portfolio-grade showcase.

## What It Demonstrates
- Correlated multi-asset Monte Carlo with optional historical calibration.
- Multiple VaR/ES methods (see below).
- Rolling backtesting (Kupiec + Christoffersen) and exception tracking.
- Stress testing (historical worst days + hypothetical scenarios).
- HTML report with method comparison and plots.

## VaR Methods Implemented
- `monte_carlo` (log-normal, correlated, optional historical calibration)
- `historical` (full revaluation historical simulation)
- `parametric_normal` (variance-covariance / delta-normal)
- `cornish_fisher` (skew/kurtosis adjusted)
- `filtered_historical` (EWMA volatility scaling)
- `student_t` (parametric t-distribution)
- `evt` (POT/GPD tail modeling)

The engine is extensible: add new methods in `src/monte_carlo_var/risk/methods.py` and register them in `src/monte_carlo_var/engine.py`.

## Quickstart
```bash
pip install -e .
mcvar --config data/sample_portfolio.json
```
Outputs land in `outputs/` by default.

Run a subset of methods:
```bash
mcvar --config data/sample_portfolio.json --methods historical,parametric_normal
```

## Configuration
See `data/sample_portfolio.json` for a full example.

Key fields:
- `portfolio.assets`: list of assets with `ticker`, `price`, `volatility`, `quantity`, and optional `drift`.
- `portfolio.correlation`: NxN matrix matching the asset count.
- `simulation`: `confidence`, `horizon_days`, `simulations`, `seed`, `trading_days`.
- `data`: `prices_path`, `date_column`, `return_type`.
- `methods`: list of VaR method configs (name + params).
- `backtest`: rolling window backtest config.
- `stress`: historical worst days and hypothetical scenarios.
- `report`: HTML report toggle.
- `output`: `directory`, `prefix`.

## Outputs
- `*_methods.json`: method summary.
- `*_results.json`: per-method VaR/ES summary.
- `*_pnl.csv`: per-method PnL series (where applicable).
- `*_pnl.png`: per-method histogram.
- `*_backtest.json` + `*_backtest_series.csv`.
- `*_stress.csv`.
- `*_report.html`.

## Project Layout
- `src/monte_carlo_var/`: core package.
- `data/`: sample config + synthetic prices.
- `tests/`: unit tests.
- `legacy/`: original scripts preserved for reference.

## Notes
- `cornish_fisher` ES is computed with a normal approximation around the adjusted quantile.
- `evt` ES is only defined for shape < 1.
