"""Fishery environment adapted to the core bilevel optimization framework.

This script demonstrates how to use the core/ framework with the legacy
fishery environment, implementing:
- MechanismContext: ContextSchema for regulatory parameters
- FisheryContextWrapper: Wrapper injecting mechanism signals into env
- MechanismOptimizer: ES-based meta optimizer (outer loop)
- FisheryOptimizer: PPO-based child optimizer (inner loop)
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any

from contexts.mechanism import MechanismParameters
from core.world.base import World
from examples.config import ES_CONFIG


# Add function to get config from yaml in the BaseOptimizerConfi
def load_config(path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_bilevel_optimization(config: dict[str, Any]) -> MechanismParameters:
    """Run bilevel optimization using the core framework.

    Args:
        config: Configuration dictionary loaded from YAML file with sections:
            - bilevel: outer_iterations, random_seed
            - es: population_size, sigma, lr_mean, lr_sigma
            - ppo: inner_iterations
            - environment: num_fishermen
            - evaluation: episodes, sustainability_weight, sustainability_threshold

    Returns:
        Best mechanism parameters found
    """
    from optimizers.ppo.training import initialize_ray, shutdown_ray

    # Extract config values
    bilevel_cfg = config.get("bilevel", {})
    es_cfg = config.get("es", {})
    ppo_cfg = config.get("ppo", {})
    env_cfg = config.get("environment", {})
    eval_cfg = config.get("evaluation", {})

    outer_iterations = bilevel_cfg.get("outer_iterations", 10)
    random_seed = bilevel_cfg.get("random_seed", 0)
    population_size = es_cfg.get("population_size", 16)
    inner_iterations = ppo_cfg.get("inner_iterations", 100)
    num_fishermen = env_cfg.get("num_fishermen", 3)
    eval_episodes = eval_cfg.get("episodes", 5)
    sustainability_weight = eval_cfg.get("sustainability_weight", 5.0)
    sustainability_threshold = eval_cfg.get("sustainability_threshold", 0.1)

    initialize_ray()

    try:
        # Create shared world
        world = World()

        # Build child optimizer (inner loop)
        child_cfg = FisheryOptimizerConfig(
            num_fishermen=num_fishermen,
            inner_iterations=inner_iterations,
            eval_episodes=eval_episodes,
            sustainability_weight=sustainability_weight,
            sustainability_threshold=sustainability_threshold,
        )
        child_opt = child_cfg.build_optimizer()
        child_opt.set_id("fishery_optimizer")
        world.register_optimizer(child_opt)

        # Build meta optimizer (outer loop)
        meta_cfg = MechanismOptimizerConfig(
            population_size=population_size,
            sigma=es_cfg.get("sigma", ES_CONFIG["sigma"]),
            mean_lr=es_cfg.get("lr_mean", ES_CONFIG["lr_mean"]),
            sigma_lr=es_cfg.get("lr_sigma", ES_CONFIG["lr_sigma"]),
            random_seed=random_seed,
        )
        meta_opt = meta_cfg.build_optimizer()
        meta_opt.set_id("mechanism_optimizer")
        meta_opt.set_downstream(child_opt)
        child_opt.set_upstream(meta_opt)
        world.register_optimizer(meta_opt)

        print("Starting bilevel optimization:")
        print(f"  Outer iterations: {outer_iterations}")
        print(f"  Population size: {population_size}")
        print(f"  Inner iterations: {inner_iterations}")
        print(f"  Fishermen: {num_fishermen}")
        print()

        # Run outer loop
        for gen in range(outer_iterations):
            print(f"\n=== Outer Iteration {gen + 1}/{outer_iterations} ===")
            meta_opt.run(world)

        # Get best result
        best_mechanism = meta_opt.get_best_mechanism()
        print(f"\nBest mechanism found: {best_mechanism}")
        print(f"Best fitness: {meta_opt.es.best_fitness:.4f}")

        return best_mechanism

    finally:
        shutdown_ray()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fishery bilevel optimization")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    best = run_bilevel_optimization(config)
    print(f"\nFinal result: {best}")
