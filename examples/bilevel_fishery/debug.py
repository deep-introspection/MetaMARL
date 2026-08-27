"""Bilevel fishery experiment with the typed metrics/reporting stack.

The outer Evolution Strategies optimizer searches ``fixed_quota`` and
``restoration_subsidy``; the inner APPO optimizer trains the fishers against
each candidate. Every level logs a typed ``MetricSchema`` and renders the
queries of :mod:`examples.bilevel_fishery.queries` through the configured
reporter (Weights & Biases by default, CSV with ``--reporter csv``).

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \\
        --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20

Full configuration: the defaults.
"""

import argparse
import logging

import numpy as np
import ray
from gymnasium import spaces

from core.adaptors.ray.schema import RaySchema
from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.es.schema import ESSchema
from core.reporting.config import ReporterConfig
from core.reporting.csv import CSVConfig
from core.reporting.wandb import WandbConfig
from examples.bilevel_fishery import queries
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanismSpace
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from examples.bilevel_fishery.regulated_env_shaefer import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

logger = logging.getLogger(__name__)

K = 5_000.0  # carrying capacity, shared by the regulated and regulator envs
OPTIMIZED = ("fixed_quota", "restoration_subsidy")


def build_reporter(args: argparse.Namespace) -> ReporterConfig:
    if args.reporter == "csv":
        return CSVConfig(project=args.project, output_dir=args.output_dir)
    return WandbConfig(
        project=args.project,
        x_disable_stats=True,
        x_disable_meta=True,
        quiet=True,
        max_end_of_run_summary_metrics=0,
        max_end_of_run_history_metrics=0,
    )


def build_mechanism_space() -> FisheryMechanismSpace:
    return FisheryMechanismSpace(
        optimize_params=list(OPTIMIZED),
        default_fixed_quota=0.56224,
        default_max_demand_frac=1.0,
        default_restoration_subsidy=0.10,
        default_fine_amount=0.00,
        default_risk_penalty_scale=0.0,
        default_risk_penalty_power=1.0,
    )


def build_config(args: argparse.Namespace) -> BilevelConfig:
    space = build_mechanism_space()
    obs_dim = 3 + space.full_dimension

    return (
        BilevelConfig()
        .world(world_name="fishery_world")
        .reporter(config=build_reporter(args))
        .mechanism(space=space)
        .training(outer_iters=args.outer_iters)
        .ray(
            device="cpu",
            num_cpus=args.num_cpus,
            omp_threads=1,
            logging_level="ERROR",
            runtime_env={
                "excludes": [
                    "wandb/",
                    ".git/",
                    ".venv/",
                    "__pycache__/",
                    "ray_results/",
                    "runs/",
                ]
            },
        )
        .outer(
            ESConfig()
            .training(
                sigma=0.15,
                mean_lr=0.10,
                sigma_decay=1.0,
                sigma_lr=0.0,
                min_sigma=0.15,
                max_sigma=0.15,
            )
            .environment(
                env=FisheryRegulatorEnv,
                env_config={
                    "ecology_cfg": {
                        "sustainability_weight": 2,
                        "sustainability_threshold": 0.20,
                        "K": K,
                    },
                },
                horizon=args.horizon,
                train_iters=args.train_iters,
            )
            .debugging(seed=42, num_seeds=1)
            .reporting(
                schema=ESSchema,
                queries=queries.ES_QUERIES
                + queries.es_parameter_queries(OPTIMIZED)
                + queries.es_candidate_fitness_queries(args.num_candidates)
                + queries.es_parameter_fitness_queries(args.num_candidates, OPTIMIZED),
            )
        )
        .inner(
            APPOptimizerConfig()
            .resources(num_cpus_for_main_process=1)
            .framework(framework="torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(
                env=FisheryRegulatedEnv,
                env_config={
                    "ecology_cfg": {
                        # Pella-Tomlinson / Schaefer single-stock dynamics
                        "r": 0.3,
                        "K": K,
                        "p": 1.0,
                        "B0": 4_000,
                        "fish_init": 4_000,
                        # env stochasticity
                        "sigma": 0.02,
                        "initial_stock_log_sigma": 0.05,
                        "unregulated_f_multiplier": 2.0,
                        "collapse_stock_frac": 0.20,
                        "collapse_transition_width": 0.03,
                        "quota_transition_width": 0.05,
                        "harvest_transition_width": 0.005,
                        "violation_transition_width": 0.03,
                        # restorative
                        "restoration_effectiveness": 0.02,
                        "restoration_effort_cost": 0.25,
                    },
                    "seed": 0,
                },
                horizon=args.horizon,
                disable_env_checking=False,
                schema=FisheryMetricSchema,
                queries=queries.FISHERY_ENV_QUERIES
                + queries.fishery_agent_queries("utilizer:0"),
            )
            .env_runners(
                num_env_runners=0,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
                num_envs_per_env_runner=args.num_candidates,
                rollout_fragment_length=args.horizon,
                batch_mode="truncate_episodes",
                max_requests_in_flight_per_env_runner=1,
            )
            .learners(num_learners=0, num_gpus_per_learner=0)
            .callbacks(
                on_episode_created=tag_episode_with_env_idx,
                on_episode_end=log_and_report_episode_metrics,
            )
            .training(
                vtrace=True,
                circular_buffer_num_batches=4,
                circular_buffer_iterations_per_batch=1,
                broadcast_interval=1,
                timeout_s_sampler_manager=10,
                timeout_s_aggregator_manager=10,
                gamma=0.99,
                lr=0.001,
                train_batch_size_per_learner=args.horizon,
                minibatch_size=args.horizon,
                num_epochs=1,
                entropy_coeff=0.001,
                grad_clip=40.0,
            )
            .evaluation(
                evaluation_interval=1,
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
                evaluation_num_env_runners=0,
                evaluation_parallel_to_training=False,
                evaluation_config={
                    "explore": False,
                    "rollout_fragment_length": args.horizon,
                    "batch_mode": "complete_episodes",
                    "max_requests_in_flight_per_env_runner": 1,
                },
                base_seed=42,
                num_seeds=args.num_eval_seeds,
            )
            .agents(
                {
                    "utilizer": {
                        "count": args.num_agents,
                        "policy": "fisher_policy",
                        "observation_space": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                        ),
                        "action_space": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                        ),
                    }
                }
            )
            .fault_tolerance(restart_failed_env_runners=False)
            .debugging(seed=42, num_seeds=1)
            .reporting(
                min_time_s_per_iteration=0,
                min_sample_timesteps_per_iteration=0,
                min_train_timesteps_per_iteration=0,
                schema=RaySchema,
                queries=queries.RAY_QUERIES,
            )
        )
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outer-iters", type=int, default=1000, help="ES generations")
    parser.add_argument(
        "--train-iters", type=int, default=50, help="APPO iterations per candidate"
    )
    parser.add_argument(
        "--horizon", type=int, default=100, help="episode length (steps)"
    )
    parser.add_argument("--num-agents", type=int, default=10, help="number of fishers")
    parser.add_argument(
        "--num-candidates", type=int, default=4, help="ES population = envs per runner"
    )
    parser.add_argument("--num-eval-seeds", type=int, default=3)
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--reporter", choices=("wandb", "csv"), default="wandb")
    parser.add_argument("--project", default="bilevel", help="reporter project name")
    parser.add_argument(
        "--output-dir", default="results", help="CSV reporter output directory"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args(argv)
    ray.shutdown()
    cfg = build_config(args)
    bilevel_opt = cfg.build_optimizer()
    try:
        result = bilevel_opt.run()
        logger.info("[debug] finished | best_fitness=%.4f", result["best_fitness"])
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
