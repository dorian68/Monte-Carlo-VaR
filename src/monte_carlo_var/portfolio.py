from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import AssetConfig


@dataclass(frozen=True)
class Portfolio:
    assets: list[AssetConfig]

    def tickers(self) -> list[str]:
        return [asset.ticker for asset in self.assets]

    def prices(self) -> np.ndarray:
        return np.array([asset.price for asset in self.assets], dtype=float)

    def quantities(self) -> np.ndarray:
        return np.array([asset.quantity for asset in self.assets], dtype=float)

    def vols(self) -> np.ndarray:
        return np.array([asset.volatility for asset in self.assets], dtype=float)

    def drifts(self) -> np.ndarray:
        return np.array([asset.drift for asset in self.assets], dtype=float)

    def value(self) -> float:
        prices = self.prices()
        quantities = self.quantities()
        return float(np.sum(prices * quantities))

    def weights(self) -> np.ndarray:
        values = self.prices() * self.quantities()
        total = np.sum(values)
        if total <= 0:
            raise ValueError("Portfolio total value must be positive.")
        return values / total
