"""Bilevel fishery experiment — APPO with frozen (non-learning) policy.

This script is used to verify the *environmental dynamics* independently of
policy learning.  It runs APPO with a :class:`FreezePolicyCallback` that
saves the initial policy weights and restores them after every training
iteration, effectively preventing any learning.

- **Outer loop**: ES with ``outer_iters=100``.  Only one inner training
  iteration per ES candidate (``train_iters=1``).
- **Inner loop**: APPO with learning rate ``lr=0.00`` and a
  :class:`FreezePolicyCallback` that snapshots and restores weights so the
  policy stays at its random initialisation.
- **Mechanism**: V0 :class:`FisheryMechanismSpace`, all parameters fixed at
  default values (high quotas, no bans).
- **No W&B reporting**.

Usage
-----
::

    uv run python -m examples.bilevel_fishery.main_appo_no_learning

Notes
-----
This script executes the experiment at module-import time.
"""

import numpy as np
import ray
from gymnasium import spaces

# core optimizers
from core.callbacks import tag_episode_with_env_idx
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.appo.config import APPOptimizerConfig

# Fishery-specific objects
from examples.bilevel_fishery.mechanism import FisheryMechanismSpace
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

# TODO the default mechanism config and fisherman, and observation spaces and action spaces part of config
# TODO where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
# TODO num_fisherman
# TODO wire up the evaluation cfg
# TODO seeding API
# TODO experimentation helpers
# TODO review ray configz

ray.shutdown()

# TODO move this to the config !
# Register custom MPS model
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork

ModelCatalog.register_custom_model("mps_fcnet", MPSFullyConnectedNetwork)


class FreezePolicyCallback(DefaultCallbacks):
    """RLlib callback that prevents any policy learning during training.

    On the first call to :meth:`on_algorithm_init` the full algorithm state
    (weights + optimiser state) is snapshotted.  After every subsequent train
    iteration :meth:`on_train_result` restores that snapshot, so the policy
    parameters remain at their random initialisation throughout the run.

    This is useful for auditing the environment dynamics in isolation: you can
    confirm that reward and fish-stock trajectories behave correctly *before*
    policy learning complicates the signal.
    """

    def on_algorithm_init(self, *, algorithm, **kwargs) -> None:
        """Snapshot the initial algorithm state immediately after construction.

        Parameters
        ----------
        algorithm : ray.rllib.algorithms.algorithm.Algorithm
            The freshly initialised RLlib algorithm instance.
        **kwargs
            Additional keyword arguments forwarded by RLlib (ignored).
        """
        self._frozen_state = algorithm.get_state()

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        """Restore the frozen state after every training step.

        Overwrites the current algorithm state with the snapshotted weights,
        effectively undoing any gradient updates performed during the iteration.
        Also sets ``result["info"]["learning_frozen"] = True`` so the flag is
        visible in the training logs.

        Parameters
        ----------
        algorithm : ray.rllib.algorithms.algorithm.Algorithm
            The RLlib algorithm instance.
        result : dict
            The training result dictionary (mutated in place to add the
            ``learning_frozen`` flag).
        **kwargs
            Additional keyword arguments forwarded by RLlib (ignored).
        """
        algorithm.set_state(self._frozen_state)
        result.setdefault("info", {})["learning_frozen"] = True


MIN_HORIZON = 50
MIN_FRAGMENT = 25
MIN_ENVS = 16
MIN_TRAIN_BATCH = MIN_HORIZON * MIN_ENVS

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .mechanism(
        space=FisheryMechanismSpace(
            max_fine=10.0,
            max_ban=200,
            default_fixed_quota=1.0,
            default_prop_quota=1.0,
            default_min_stock=0.10,
            default_fine_amount=0.5,
            default_ban_period=0,
            default_catch_prob=1.0,
        ),
    )
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
        .training(  # TODO dimension inferred from mechanism ?
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
            horizon=MIN_HORIZON,
            train_iters=1,  # TODO implement early stop for plateau
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
        # TODO fix this its using old api stack
        # .model(custom_model="mps_fcnet")
        # TODO use the new api stack and better custom model integration
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .environment(
            env=FisheryRegulatedEnv,
            env_config={
                "ecology_cfg": {
                    "algae_init": 1.0,
                    "fish_init": 1.0,
                    "max_fish": 5.0,
                    "max_algae": 5.0,
                    "alpha": 0.5,
                    "beta": 0.1,
                    "delta": 0.2,
                    "gamma": 0.4,
                    "dt": 0.01,
                },
                "seed": 0,
            },
            horizon=MIN_HORIZON,
            disable_env_checking=False,
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=16,  # batch evaluated mechanism or population size for ES 16
            rollout_fragment_length=MIN_FRAGMENT,  # must be same as env horizon 200
            batch_mode="truncate_episodes",
        )
        .learners(num_learners=1, num_gpus_per_learner=0)
        .callbacks(
            on_episode_created=tag_episode_with_env_idx,  # New API stack
            callbacks_class=FreezePolicyCallback,
        )
        .training(
            vtrace=True,
            circular_buffer_num_batches=1,  # TODO review
            circular_buffer_iterations_per_batch=1,  # TODO review
            # minibatch_buffer_size=200,
            broadcast_interval=10_000,
            # learner_queue_size=64,
            # learner_queue_timeout=300,
            timeout_s_sampler_manager=300,
            timeout_s_aggregator_manager=300,
            gamma=0.99,
            lr=0.00,
            train_batch_size=MIN_TRAIN_BATCH,  # 3200
            minibatch_size=MIN_TRAIN_BATCH,  # 512
            entropy_coeff=0.0,
            # entropy_coeff_schedule=[
            #     [0, 0.01],
            #     [200_000, 0.001],
            #     [1_000_000, 0.0],
            # ],
            grad_clip=0.0,
            # lr_schedule=[
            #     [0, 1e-3],
            #     [300_000, 3e-4],
            #     [1_000_000, 1e-4],
            # ]
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=MIN_HORIZON
            * 16,  # rollout_fragment_length X num_episodes
            evaluation_duration_unit="timesteps",
            evaluation_num_env_runners=1,
            # evaluation_parallel_to_training=False,  # keep it simple/deterministic
            evaluation_config={
                "explore": False,  # greedy eval actions
                "seed": 42,
                "num_envs_per_env_runner": 16,  # same as training
                "rollout_fragment_length": MIN_HORIZON,  # same as training
                "batch_mode": "complete_episodes",  # same as training
                "minibatch_size": None,
            },
        )
        .agents(
            {
                "fisher": {
                    "count": 3,
                    "policy": "fisher_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            5 + FisheryMechanismSpace().full_dimension,
                        ),  # fish and alage #mechanism conditioned-RL
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
    )
)

# TODO
# custom_evaluation_function

# run the experiment
bilevel_opt = bilevel_opt_cfg.build_optimizer()
bilevel_opt.run()

# TODO add this after run done
ray.shutdown()
