"""
Parameterized bilevel fishery experiment launcher (hypothesis-variant grid).

Clone of ``debug.py`` (baseline config from feat/fresh-water-rework, tip
1498456) where the experiment axes are exposed on the CLI instead of being
edited in place. Everything not exposed as a flag is IDENTICAL to debug.py.

Axes
----
--optimize-params : which mechanism parameters the outer ES optimizes
                    (subset of {fixed_quota, min_demand_frac, max_demand_frac}).
                    Baseline: min_demand_frac (1-D).
--population      : ES population size, implemented as
                    num_envs_per_env_runner = population (1 inner seed).
                    Must stay even for antithetic pairs when > 1
                    (break_symmetry=False). Baseline: 1.
--outer-iters     : number of outer ES generations. Baseline: 1000.
--train-iters     : inner APPO iterations per generation. Baseline: 50.
--seed            : outer ES seed and inner RLlib seed. Baseline: 42.
--env-seed        : environment noise seed. Baseline: 0.
--run-label       : short label baked into the wandb run name via world_name
                    (run name becomes "bilevel-fishery_<label>_<uuid8>").

When to use
-----------
Running the {population} x {mechanism dimensionality} hypothesis grid on top
of the baseline model without mutating debug.py between runs. Each variant is
one CLI invocation, so the exact configuration is preserved in the shell
history, the run log, and the wandb run name.

Examples
--------
Baseline replication (identical to debug.py)::

    PYTHONPATH=. python examples/bilevel_fishery/exp_run.py --run-label pop1-1d-s42

Population 16, 2-D mechanism, 300 generations::

    PYTHONPATH=. python examples/bilevel_fishery/exp_run.py \\
        --population 16 --optimize-params min_demand_frac fixed_quota \\
        --outer-iters 300 --run-label pop16-2d-s42
"""

import argparse

import numpy as np
import ray
from gymnasium import spaces

from core.callbacks import tag_episode_with_env_idx
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.appo.config import APPOptimizerConfig

from examples.bilevel_fishery.mechanism_v1 import FisheryMechanismSpace
from examples.bilevel_fishery.regulated_env_shaefer import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bilevel fishery experiment (variant grid over debug.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--optimize-params",
        nargs="+",
        default=["min_demand_frac"],
        choices=["fixed_quota", "min_demand_frac", "max_demand_frac"],
        help="Mechanism parameters optimized by the outer ES",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=1,
        help="ES population size (= num_envs_per_env_runner; even if > 1)",
    )
    parser.add_argument(
        "--outer-iters", type=int, default=1000, help="Outer ES generations"
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=50,
        help="Inner APPO iterations per generation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Outer ES + inner RLlib seed"
    )
    parser.add_argument(
        "--env-seed", type=int, default=0, help="Environment noise seed"
    )
    parser.add_argument(
        "--run-label",
        type=str,
        required=True,
        help="Short label for the wandb run name, e.g. pop16-2d-s42",
    )
    args = parser.parse_args()

    if args.population > 1 and args.population % 2 != 0:
        parser.error(
            "--population must be even when > 1 (antithetic sampling pairs)"
        )
    return args


def build_config(args: argparse.Namespace) -> BilevelConfig:
    return (
        BilevelConfig()
        .world(world_name=f"fishery_{args.run_label}")
        .reporting(
            reporter="wandb",
            project_name="bilevel",
            settings_dict={
                "x_disable_stats": True,
                "x_disable_meta": True,
                "quiet": True,
                "max_end_of_run_summary_metrics": 0,
                "max_end_of_run_history_metrics": 0,
            },
        )
        .mechanism(
            space=FisheryMechanismSpace(
                optimize_params=list(args.optimize_params),
                # "extremely restrictive quota" baseline defaults (debug.py)
                default_fixed_quota=0.90,
                default_min_demand_frac=0.05,
                default_max_demand_frac=1.0,
                default_fine_amount=0.10,
                default_risk_penalty_scale=1.0,
                default_risk_penalty_power=2.0,
            )
        )
        .training(outer_iters=args.outer_iters)
        .ray(
            device="cpu",
            num_cpus=4,
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
                sigma_lr=0.02,
                min_sigma=0.01,
                max_sigma=0.25,
            )
            .environment(
                env=FisheryRegulatorEnv,
                env_config={
                    "ecology_cfg": {
                        "sus_weight": 1.0,
                        "sus_threshold": 0.1,
                        "K": 5_000,  # HAS to match environment K
                    },
                },
                horizon=100,
                train_iters=args.train_iters,
            )
            .debugging(
                seed=args.seed,
                num_seeds=10,
            )
        )
        .inner(
            APPOptimizerConfig()
            .resources(
                num_cpus_for_main_process=1,
            )
            .framework(
                framework="torch",
            )
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
                        "K": 5_000,
                        "p": 1.0,
                        "sigma": 0.05,
                        "B0": 2_500,
                        # Optional compatibility alias used by reset if present
                        "fish_init": 2_500,
                    },
                    "seed": args.env_seed,
                },
                horizon=100,
                disable_env_checking=False,
            )
            .env_runners(
                num_env_runners=0,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
                # ES population size = num_envs_per_env_runner // len(inner
                # seeds). With 1 inner seed -> population = num_envs. Must stay
                # even for antithetic pairs while break_symmetry=False.
                num_envs_per_env_runner=args.population,
                rollout_fragment_length=100,
                batch_mode="truncate_episodes",
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=0,
            )
            .callbacks(
                on_episode_created=tag_episode_with_env_idx,
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
                train_batch_size_per_learner=100,
                minibatch_size=100,
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
                    "rollout_fragment_length": 100,
                    "batch_mode": "complete_episodes",
                },
            )
            .agents(
                {
                    "utilizer": {
                        "count": 5,
                        "policy": "fisher_policy",
                        "observation_space": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(
                                3 + FisheryMechanismSpace().full_dimension,
                            ),
                            dtype=np.float32,
                        ),
                        "action_space": spaces.Box(
                            # Fraction of the agent's maximum pull capacity
                            # (see debug.py for the full interpretation table).
                            low=0.0,
                            high=1,
                            shape=(1,),
                            dtype=np.float32,
                        ),
                    }
                }
            )
            .fault_tolerance(
                restart_failed_env_runners=False,
            )
            .debugging(
                seed=args.seed,
                num_seeds=1,
            )
            .reporting(
                min_time_s_per_iteration=0,
                min_sample_timesteps_per_iteration=0,
                min_train_timesteps_per_iteration=0,
            )
        )
    )


if __name__ == "__main__":
    cli_args = parse_args()

    ray.shutdown()
    bilevel_opt = build_config(cli_args).build_optimizer()
    bilevel_opt.run()
    ray.shutdown()
