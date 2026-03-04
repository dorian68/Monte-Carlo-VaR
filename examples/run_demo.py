from pathlib import Path

from monte_carlo_var.config import load_config
from monte_carlo_var.engine import run_var_methods
from monte_carlo_var.reporting import save_methods_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "data" / "sample_portfolio.json"
    cfg = load_config(config_path)
    results = run_var_methods(cfg)

    out_dir = root / cfg.output.directory
    out_dir.mkdir(parents=True, exist_ok=True)

    save_methods_json(results, out_dir / f"{cfg.output.prefix}_methods.json")
    print("Demo complete. Outputs:", out_dir.resolve())


if __name__ == "__main__":
    main()
