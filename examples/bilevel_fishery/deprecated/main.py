import argparse
import logging
from pathlib import Path

import ray
import yaml

from examples.bilevel_fishery.bilevel import BilevelConfigLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser("Bilevel Fishery Experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (relative to project root or absolute)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save visualization plots (default: True)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable visualization plots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    ray.shutdown()

    output_dir = None if args.no_plots else args.output_dir
    cfg = BilevelConfigLoader.from_yaml(args.config, output_dir=output_dir)
    optimizer = cfg.build_optimizer()
    results = optimizer.run()

    ray.shutdown()

    if not args.no_plots and results.get("best_trajectory"):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from core.reporting.utils.ray_old_api_stack import plot_combined_trial_analysis

        # Load scaling values from config
        with open(config_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        scaling_cfg = raw_cfg.get("mechanism", {}).get("scaling", {})
        max_fine = scaling_cfg.get("max_fine", 5.0)
        max_ban = scaling_cfg.get("max_ban", 50)

        # Load ecology config for sustainability threshold
        outer_ecology = (
            raw_cfg.get("outer", {})
            .get("environment", {})
            .get("env_config", {})
            .get("ecology_cfg", {})
        )
        sus_threshold = outer_ecology.get("sus_threshold", 0.1)
        max_fish = outer_ecology.get("max_fish", 2.0)
        raw_sus_threshold = sus_threshold * max_fish

        mechanism_params = None
        if results.get("best_mechanism") is not None and hasattr(
            optimizer, "mechanism_space"
        ):
            # Decode using mechanism space to get actual parameter values
            best_mech = optimizer.mechanism_space.decode(results["best_mechanism"])
            mechanism_params = {
                "min_stock": best_mech.min_stock,
                "fine_amount": best_mech.fine_amount,
                "catch_prob": best_mech.catch_prob,
            }

        save_path = output_dir / "trial_analysis.png"
        plot_combined_trial_analysis(
            results["best_trajectory"],
            mechanism_params=mechanism_params,
            sustainability_threshold=raw_sus_threshold,
            title=f"Best Mechanism (fitness={results['best_fitness']:.4f})",
            save_path=str(save_path),
        )
        logger.info(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
