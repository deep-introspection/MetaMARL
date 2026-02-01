import argparse
from pathlib import Path

import ray

from examples.bilevel_fishery.bilevel import BilevelConfigLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser("Bilevel Fishery Experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (relative to project root or absolute)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    ray.shutdown()

    cfg = BilevelConfigLoader.from_yaml(args.config)
    optimizer = cfg.build_optimizer()
    optimizer.run()

    ray.shutdown()


if __name__ == "__main__":
    main()
