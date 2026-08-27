import numpy as np
import ray
from gymnasium import spaces

from core.callbacks import tag_episode_with_env_idx
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanismSpace
from examples.bilevel_fishery.regulated_env_shaefer import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

ray.shutdown()

from ray.rllib.models import ModelCatalog

from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork

ModelCatalog.register_custom_model("mps_fcnet", MPSFullyConnectedNetwork)

mechanism_space = FisheryMechanismSpace(
    default_fixed_quota=0.25,
    default_prop_quota=0.25,
    default_min_stock=0.40,
    default_target_stock=0.6,
    default_fine_amount=0.5,
    default_risk_penalty_scale=0.8,
    default_risk_penalty_power=2.0,
)

fishery_ecology_cfg = {
    # Pella-Tomlinson / Schaefer single-stock dynamics
    "r": 0.3,
    "K": 5.0,
    "p": 1.0,
    "sigma": 0.05,
    "B0": 2.5,
    # Optional compatibility alias used by reset if present
    "fish_init": 2.5,
}

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
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
    .mechanism(space=mechanism_space)
    .training(outer_iters=100)
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
            sigma=0.5,
            mean_lr=0.2,
            sigma_lr=0.05,
            min_sigma=0.01,
            max_sigma=0.6,
        )
        .environment(
            env=FisheryRegulatorEnv,
            env_config={
                "ecology_cfg": {
                    "sus_weight": 1.0,
                    "sus_threshold": 0.1,
                },
            },
            horizon=500,
            train_iters=100,
        )
        .debugging(seed=42, num_seeds=1)
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
                "ecology_cfg": fishery_ecology_cfg,
                "seed": 0,
            },
            horizon=500,
            disable_env_checking=False,
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=4,
            rollout_fragment_length=500,
            batch_mode="truncate_episodes",
        )
        .learners(num_learners=0, num_gpus_per_learner=0)
        .callbacks(on_episode_created=tag_episode_with_env_idx)
        .training(
            vtrace=True,
            circular_buffer_num_batches=4,
            circular_buffer_iterations_per_batch=1,
            broadcast_interval=1,
            timeout_s_sampler_manager=10,
            timeout_s_aggregator_manager=10,
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=500,
            minibatch_size=500,
            num_epochs=1,
            entropy_coeff=0.001,
            grad_clip=40.0,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=4,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,
                "rollout_fragment_length": 500,
                "batch_mode": "complete_episodes",
            },
        )
        .agents(
            {
                "fisher": {
                    "count": 1,
                    "policy": "fisher_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3 + mechanism_space.full_dimension,),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(1,),
                        dtype=np.float32,
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
        )
    )
)

bilevel_opt = bilevel_opt_cfg.build_optimizer()
bilevel_opt.run()

ray.shutdown()
