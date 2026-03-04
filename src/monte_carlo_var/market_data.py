from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_price_data(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in price data.")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).set_index(date_column)

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("Price data contains no numeric asset columns.")
    return numeric_df


def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    if return_type not in {"log", "simple"}:
        raise ValueError("return_type must be 'log' or 'simple'.")

    if return_type == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna()
