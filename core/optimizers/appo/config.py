from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.appo import APPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class APPOptimizerConfig(RayOptimizerConfig):
    """Configuration for the Asynchronous Proximal Policy Optimization inner optimizer.

    Thin subclass of :class:`~core.adaptors.ray.optimizer_config.RayOptimizerConfig`
    that binds the Ray RLlib ``APPO`` algorithm class as the concrete backend.  All
    builder methods (``environment``, ``training``, ``env_runners``, ``multi_agent``,
    etc.) are inherited from ``RayOptimizerConfig`` and delegate directly to the
    underlying ``AlgorithmConfig`` via the deferred ``_cfg_ops`` queue.

    APPO extends PPO with asynchronous data collection: env-runners generate
    rollouts concurrently while the learner updates policy weights, which improves
    hardware utilisation at the cost of slightly stale samples.  See:
    Liang et al. (2018) "RLlib: Abstractions for Distributed Reinforcement Learning"
    (https://arxiv.org/abs/1712.09381).

    Attributes
    ----------
    algo_class : type[Algorithm]
        Ray RLlib algorithm class used to build the underlying trainer.  Fixed to
        ``APPO`` for this config subclass.

    Examples
    --------
    >>> cfg = (
    ...     APPOptimizerConfig()
    ...     .environment(env=FisherEnv, train_iters=200)
    ...     .training(lr=5e-4, gamma=0.99)
    ...     .env_runners(num_env_runners=4)
    ... )
    >>> optimizer = cfg.build_optimizer(world=world_handle, world_name="run_01",
    ...                                  reporting=reporter_handle)
    """

    algo_class: Algorithm = APPO
