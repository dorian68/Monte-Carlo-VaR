from monte_carlo_var.config import AssetConfig, PortfolioConfig, RunConfig, SimulationConfig
from monte_carlo_var.simulation import run_monte_carlo


def test_run_monte_carlo_shapes():
    assets = [AssetConfig(ticker="TEST", price=100.0, volatility=0.2, quantity=10, drift=0.05)]
    portfolio = PortfolioConfig(assets=assets, correlation=[[1.0]])
    simulation = SimulationConfig(confidence=0.95, horizon_days=10, simulations=500, seed=7)
    cfg = RunConfig(portfolio=portfolio, simulation=simulation)

    result = run_monte_carlo(cfg)

    assert result.pnl.shape == (500,)
    assert result.portfolio_value == 100.0 * 10
    assert result.var >= 0.0
