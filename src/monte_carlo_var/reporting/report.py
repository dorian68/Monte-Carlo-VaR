from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..backtest import BacktestResult
from ..risk import VarResult
from ..stress import StressResult


def save_pnl_csv(pnl: np.ndarray, path: str | Path) -> None:
    df = pd.DataFrame({"pnl": pnl})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_results_json(result: VarResult, path: str | Path) -> None:
    payload = {
        "method": result.method,
        "var": result.var,
        "es": result.es,
        "confidence": result.confidence,
        "metadata": result.metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_methods_json(results: Iterable[VarResult], path: str | Path) -> None:
    payload = {
        result.method: {
            "var": result.var,
            "es": result.es,
            "confidence": result.confidence,
            "metadata": result.metadata,
        }
        for result in results
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_backtest_json(result: BacktestResult, path: str | Path) -> None:
    payload = {
        "exceptions": result.exceptions,
        "observations": result.observations,
        "exception_rate": result.exception_rate,
        "lr_uc": result.lr_uc,
        "p_value_uc": result.p_value_uc,
        "lr_ind": result.lr_ind,
        "p_value_ind": result.p_value_ind,
        "lr_cc": result.lr_cc,
        "p_value_cc": result.p_value_cc,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_stress_csv(results: Iterable[StressResult], path: str | Path) -> None:
    df = pd.DataFrame(
        [{"scenario": r.name, "portfolio_return": r.portfolio_return, "pnl": r.pnl} for r in results]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_pnl_histogram(
    pnl: np.ndarray,
    var: float,
    es: float,
    path: str | Path,
    title: str = "PnL Distribution",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnl, bins=60, alpha=0.75, color="#2c7fb8", edgecolor="#ffffff")
    ax.axvline(-var, color="#d7301f", linestyle="--", label=f"VaR = {var:,.0f}")
    ax.axvline(-es, color="#7a0177", linestyle=":", label=f"ES = {es:,.0f}")
    ax.set_title(title)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_html_report(
    results: Iterable[VarResult],
    plots: dict[str, str],
    backtest: BacktestResult | None,
    stress: Iterable[StressResult] | None,
    path: str | Path,
) -> None:
    rows = []
    for result in results:
        rows.append(
            f"<tr><td>{result.method}</td><td>{result.confidence:.2%}</td><td>{result.var:,.2f}</td><td>{result.es:,.2f}</td></tr>"
        )

    plot_blocks = []
    for method, plot_path in plots.items():
        plot_blocks.append(
            f"<div class='plot'><h3>{method}</h3><img src='{plot_path}' alt='{method} plot'/></div>"
        )

    backtest_block = ""
    if backtest:
        backtest_block = (
            "<h2>Backtest</h2>"
            "<table><tr><th>Exceptions</th><th>Obs</th><th>Rate</th><th>LR_uc</th><th>p_uc</th><th>LR_ind</th><th>p_ind</th><th>LR_cc</th><th>p_cc</th></tr>"
            f"<tr><td>{backtest.exceptions}</td><td>{backtest.observations}</td><td>{backtest.exception_rate:.2%}</td>"
            f"<td>{backtest.lr_uc:.3f}</td><td>{backtest.p_value_uc:.3f}</td>"
            f"<td>{backtest.lr_ind:.3f}</td><td>{backtest.p_value_ind:.3f}</td>"
            f"<td>{backtest.lr_cc:.3f}</td><td>{backtest.p_value_cc:.3f}</td></tr></table>"
        )

    stress_block = ""
    if stress:
        stress_rows = "".join(
            f"<tr><td>{item.name}</td><td>{item.portfolio_return:.2%}</td><td>{item.pnl:,.2f}</td></tr>"
            for item in stress
        )
        stress_block = (
            "<h2>Stress Tests</h2>"
            "<table><tr><th>Scenario</th><th>Portfolio Return</th><th>PnL</th></tr>"
            f"{stress_rows}</table>"
        )

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>VaR Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1b1b1b; }}
    h1, h2 {{ color: #0b3d91; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .plot img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>VaR Report</h1>
  <h2>Method Summary</h2>
  <table>
    <tr><th>Method</th><th>Confidence</th><th>VaR (loss)</th><th>ES (loss)</th></tr>
    {"".join(rows)}
  </table>

  <div class="plots">
    {"".join(plot_blocks)}
  </div>

  {backtest_block}
  {stress_block}
</body>
</html>
"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
