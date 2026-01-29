import numpy as np
from gymnasium import spaces

# core optimizers
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig

# Fishery-specific objects
from examples.bilevel_fishery.mechanism import FisheryMechnaismSpace
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

# TODO the default mechanism config and fisherman, and observation spaces and action spaces part of config
# TODO where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
# TODO num_fisherman
# TODO wire up the evaluation cfg


OBSERVATION_SPACES = spaces.Dict(
    {
        "fish": spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.float32,
        ),
        "algae": spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(1, 0),
            dtype=np.float32,
        ),
    }
)

ACTION_SPACES = spaces.Dict(
    {
        "fish_harvest": spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.float32,
        )
    }
)


bilevel_optimizer: BilevelOptimizer = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .outer(
        ESConfig()
        .training(
            dimension=5,  # TODO dimension inferred from mechanism ?
            pop_size=16,
            sigma=0.15,
            mean_lr=0.1,
            sigma_lr=0.05,
            min_sigma=1e-3,
            max_sigma=0.5,
        )
        .environment(
            env=FisheryRegulatorEnv,
            env_config={
                "mechanism_space": FisheryMechnaismSpace().from_dict(
                    cfg={
                        "fixed_quota": 0.2,  # Fixed harvest quota
                        "prop_quota": 0.1,  # Proportional quota factor
                        "min_stock": 0.1,  # Minimum stock threshold
                        "fine_amount": 1.0,  # Fine per unit over-harvest
                        "ban_period": 2,  # Periods banned after violation
                    }
                ),
                "ecology_cfg": {"sus_weight": 5.0, "sus_threshold": 0.1},
            },
            train_iters=100,
        )
    )
    .inner(
        PPOptimizerConfig()
        .environment(
            env=FisheryRegulatedEnv,
            env_config={
                "ecology_cfg": {
                    "algae_init": 1.0,
                    "fish_init": 0.5,
                    "alpha": 1.0,
                    "beta": 0.5,
                    "delta": 0.5,
                    "gamma": 1.0,
                    "dt": 0.05,
                },
            },
            horizon=200,
        )
        .framework(
            framework="torch",
        )
        .env_runners(
            num_env_runners=0,  # num_workers
        )
        .resources(num_gpus=0, num_cpus_for_driver=1, num_cpu_per_worker=1)
        .training(gramma=0.99, train_batch_size=1000, minibatch_size=128, lr=3e-4)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=5,  # eval iters
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            always_attach_evaluation_results=True,
            evaluation_config={
                "explore": False,
            },
        )
        .agents(
            {
                "fisher": {
                    "count": 5,
                    "policy": "fisher_policy",
                    "observation_space": OBSERVATION_SPACES,
                    "action_space": ACTION_SPACES,
                }
            }
        )
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .fault_tolerance(restart_failed_env_runners=False)
    )
    .training(outer_iters=10)
    .ray(
        device="cpu",
        num_cpus=8,
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
)


# run the experiment
bilevel_optimizer.run()
