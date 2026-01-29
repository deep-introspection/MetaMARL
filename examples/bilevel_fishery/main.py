# TODO Rename PPOptimizerConfig to PPOConfig
import numpy as np
from gymnasium import spaces

from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer
from core.optimizers.es.config import ESConfig

# TODO Move both th config and the optimizer into same file
from core.optimizers.ppo.config import PPOptimizerConfig
from examples.bilevel_fishery.mechanism import FisheryMechnaismSpace
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

# Default mechanism parameters
DEFAULT_MECHANISM_CONFIG = {
    "fixed_quota": 0.2,  # Fixed harvest quota
    "prop_quota": 0.1,  # Proportional quota factor
    "min_stock": 0.1,  # Minimum stock threshold
    "fine_amount": 1.0,  # Fine per unit over-harvest
    "ban_period": 2,  # Periods banned after violation
}

# Observation/action space bounds
FISHERMEN = ["f0"]

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
                    cfg=DEFAULT_MECHANISM_CONFIG
                ),
                "ecology_cfg": {"sus_weight": 5.0, "sus_threshold": 0.1},
            },
            train_iters=100,
            eval_iters=5,
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
        .training(gramma=0.99, train_batch_size=4000, minibatch_size=512, lr=3e-4)
        .multi_agent(
            policies={"fisher_policy": (None, OBSERVATION_SPACES, ACTION_SPACES, {})},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "fisher_policy",
            policies_to_train=["fisher_policy"],
        )
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .fault_tolerance(restart_failed_env_runners=False)
    )
    .training(outer_iters=10)
)


# run the experiment
bilevel_optimizer.run()
