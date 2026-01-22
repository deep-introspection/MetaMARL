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

from typing import Any, Optional

from gymnasium import Env

from config import ES_CONFIG
from core.optimizers.base import Optimizer
from core.optimizers.config import BaseOptimizerConfig
from core.world.base import World
from core.world.context import Context, ContextSchema
from core.wrappers.context_wrapper import ContextWrapper
from evaluation import evaluate_mechanism_with_metrics
from evolution_strategies import EvolutionStrategies
from mechanism import MechanismParameters, map_unit_vector_to_mechanism
from training import build_ppo_algorithm, train_algorithm


def load_config(path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Context Schemas
# =============================================================================


class MechanismContext(ContextSchema):
    """Regulatory mechanism parameters published by meta optimizer."""

    fixed_quota: float
    prop_quota: float
    min_stock: float
    fine_amount: float
    ban_period: int

    @classmethod
    def from_mechanism_params(cls, params: MechanismParameters) -> "MechanismContext":
        return cls(
            fixed_quota=params.fixed_quota,
            prop_quota=params.prop_quota,
            min_stock=params.min_stock,
            fine_amount=params.fine_amount,
            ban_period=params.ban_period,
        )

    def to_mechanism_params(self) -> MechanismParameters:
        return MechanismParameters(
            fixed_quota=self.fixed_quota,
            prop_quota=self.prop_quota,
            min_stock=self.min_stock,
            fine_amount=self.fine_amount,
            ban_period=self.ban_period,
        )


class FitnessContext(ContextSchema):
    """Fitness score published by child optimizer after evaluation."""

    objective_score: float
    mean_reward: float
    collapse_rate: float
    sustainability_penalty: float


# =============================================================================
# Context Wrapper
# =============================================================================


class FisheryContextWrapper(ContextWrapper):
    """Wrapper that injects mechanism parameters from World into FisheryEnv.

    This wrapper reads the current MechanismContext from the World and
    applies violation penalties to the reward based on quota violations.
    """

    def __init__(self, env: Env, world: World, meta_opt_id: str) -> None:
        super().__init__(env, world)
        self._meta_opt_id = meta_opt_id
        self._cached_mechanism: Optional[MechanismContext] = None

    def _get_mechanism_context(self) -> Optional[MechanismContext]:
        """Retrieve current mechanism context from world."""
        ctx_ids = self._world.get_opt_ctx_ids(opt_id=self._meta_opt_id)
        for ctx_id in ctx_ids:
            ctx = self._world.get_context(ctx_id)
            if ctx and isinstance(ctx.payload, MechanismContext):
                return ctx.payload
        return None

    def _get_violation_signal(self) -> float:
        """Return 0 - no additional penalty beyond env's built-in mechanism."""
        return 0.0

    def _get_violation_penalty(self) -> float:
        """Return 0 - penalties handled by FisheryEnvFixed directly."""
        return 0.0

    def observation(self, observation):
        """Pass through observation unchanged."""
        return observation

    def action(self, action):
        """Pass through action unchanged."""
        return action

    def get_current_mechanism(self) -> Optional[MechanismContext]:
        """Get the current mechanism context for external use."""
        return self._get_mechanism_context()


# =============================================================================
# Optimizer Configs
# =============================================================================


class MechanismOptimizerConfig(BaseOptimizerConfig):
    """Configuration for the ES-based mechanism optimizer."""

    def __init__(
        self,
        population_size: int = 16,
        sigma: float = 0.15,
        mean_lr: float = 0.1,
        sigma_lr: float = 0.05,
        random_seed: int = 0,
    ):
        super().__init__(opt_class=MechanismOptimizer)
        self.population_size = population_size
        self.sigma = sigma
        self.mean_lr = mean_lr
        self.sigma_lr = sigma_lr
        self.random_seed = random_seed


class FisheryOptimizerConfig(BaseOptimizerConfig):
    """Configuration for the PPO-based fishery optimizer."""

    def __init__(
        self,
        num_fishermen: int = 3,
        inner_iterations: int = 100,
        eval_episodes: int = 5,
        sustainability_weight: float = 5.0,
        sustainability_threshold: float = 0.1,
    ):
        super().__init__(opt_class=FisheryOptimizer)
        self.num_fishermen = num_fishermen
        self.inner_iterations = inner_iterations
        self.eval_episodes = eval_episodes
        self.sustainability_weight = sustainability_weight
        self.sustainability_threshold = sustainability_threshold


# =============================================================================
# Optimizers
# =============================================================================


class MechanismOptimizer(Optimizer):
    """Evolution Strategies optimizer for mechanism parameter search.

    This is the outer-loop (meta) optimizer that:
    1. Samples mechanism parameter candidates using ES
    2. Publishes each candidate as MechanismContext to World
    3. Runs downstream FisheryOptimizer for each candidate
    4. Collects fitness scores and updates ES distribution
    """

    def __init__(self, config: MechanismOptimizerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.es = EvolutionStrategies(
            dimension=5,
            population_size=config.population_size,
            sigma=config.sigma,
            mean_learning_rate=config.mean_lr,
            sigma_learning_rate=config.sigma_lr,
            random_seed=config.random_seed,
        )
        self._mechanism_ctx_id: Optional[str] = None

    def run(self, world: World) -> None:
        """Run one generation of ES optimization."""
        # Sample population of mechanism candidates
        population = self.es.sample_population()
        fitness_scores = []

        for i, candidate in enumerate(population):
            # Convert unit vector to mechanism parameters
            mechanism_params = map_unit_vector_to_mechanism(candidate)
            mechanism_ctx = MechanismContext.from_mechanism_params(mechanism_params)

            # Publish or update mechanism context in world
            ctx = Context(
                id=f"mechanism_{self.id}",
                opt_id=self.id,
                payload=mechanism_ctx,
            )

            if self._mechanism_ctx_id is None:
                self._mechanism_ctx_id = world.set_new_context(ctx)
            else:
                ctx.id = self._mechanism_ctx_id
                world.update_context(ctx)

            # Run downstream optimizers (FisheryOptimizer)
            for downstream_opt in self._downstream:
                downstream_opt.run(world)

                # Collect fitness from downstream optimizer's context
                fitness = self._collect_fitness(world, downstream_opt)
                fitness_scores.append(fitness)

        # Update ES distribution based on fitness scores
        self.es.update_parameters(population, fitness_scores)

        print(
            f"Generation {self.es.generation}: "
            f"best_fitness={self.es.best_fitness:.4f}, "
            f"sigma={self.es.sigma:.4f}"
        )

    def _collect_fitness(self, world: World, opt: Optimizer) -> float:
        """Collect fitness score from downstream optimizer's context."""
        ctx_ids = world.get_opt_ctx_ids(opt.id)
        for ctx_id in ctx_ids:
            ctx = world.get_context(ctx_id)
            if ctx and isinstance(ctx.payload, FitnessContext):
                return ctx.payload.objective_score
        return float("-inf")

    def evaluate(self, world: World) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def get_best_mechanism(self) -> MechanismParameters:
        """Return the best mechanism found so far."""
        return map_unit_vector_to_mechanism(self.es.best_candidate)


class FisheryOptimizer(Optimizer):
    """PPO-based optimizer for training fishermen agents.

    This is the inner-loop optimizer that:
    1. Reads mechanism parameters from World (set by MechanismOptimizer)
    2. Builds/configures PPO with current mechanism
    3. Trains agents for N iterations
    4. Evaluates and publishes FitnessContext to World
    """

    def __init__(self, config: FisheryOptimizerConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.num_fishermen = config.num_fishermen
        self.inner_iterations = config.inner_iterations
        self.eval_episodes = config.eval_episodes
        self.sustainability_weight = config.sustainability_weight
        self.sustainability_threshold = config.sustainability_threshold
        self._fitness_ctx_id: Optional[str] = None
        self._algorithm = None

    def run(self, world: World) -> None:
        """Train and evaluate with current mechanism from world."""
        # Get mechanism context from upstream optimizer
        mechanism_ctx = self._get_mechanism_from_world(world)
        if mechanism_ctx is None:
            raise RuntimeError("No MechanismContext found in world")

        mechanism_params = mechanism_ctx.to_mechanism_params()

        # Build PPO algorithm with current mechanism
        algorithm, env = build_ppo_algorithm(
            num_fishermen=self.num_fishermen,
            mechanism_params=mechanism_params,
            num_workers=0,
        )
        self._algorithm = algorithm

        # Train
        train_algorithm(algorithm, self.inner_iterations)

        # Evaluate
        metrics, _ = evaluate_mechanism_with_metrics(
            algorithm=algorithm,
            environment=env,
            num_episodes=self.eval_episodes,
            sustainability_threshold=self.sustainability_threshold,
        )

        # Compute objective score
        objective_score = metrics.objective_score(self.sustainability_weight)

        # Publish fitness context
        fitness_ctx = FitnessContext(
            objective_score=objective_score,
            mean_reward=metrics.mean_reward,
            collapse_rate=metrics.collapse_rate,
            sustainability_penalty=metrics.sustainability_penalty,
        )

        ctx = Context(
            id=f"fitness_{self.id}",
            opt_id=self.id,
            payload=fitness_ctx,
        )

        if self._fitness_ctx_id is None:
            self._fitness_ctx_id = world.set_new_context(ctx)
        else:
            ctx.id = self._fitness_ctx_id
            world.update_context(ctx)

        # Cleanup
        algorithm.stop()

    def _get_mechanism_from_world(self, world: World) -> Optional[MechanismContext]:
        """Find MechanismContext from any upstream optimizer."""
        for upstream_opt in self._upstream:
            ctx_ids = world.get_opt_ctx_ids(upstream_opt.id)
            for ctx_id in ctx_ids:
                ctx = world.get_context(ctx_id)
                if ctx and isinstance(ctx.payload, MechanismContext):
                    return ctx.payload
        return None

    def evaluate(self, world: World) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass


# =============================================================================
# Main Entry Point
# =============================================================================


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
    from training import initialize_ray, shutdown_ray

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
