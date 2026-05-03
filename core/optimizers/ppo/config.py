from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class PPOptimizerConfig(RayOptimizerConfig):
    """Configuration for the Proximal Policy Optimization inner optimizer.

    Thin subclass of :class:`~core.adaptors.ray.optimizer_config.RayOptimizerConfig`
    that binds the Ray RLlib ``PPO`` algorithm class as the concrete backend.
    All builder methods (``environment``, ``training``, ``env_runners``,
    ``multi_agent``, etc.) are inherited from ``RayOptimizerConfig`` and
    delegate directly to the underlying ``AlgorithmConfig`` via the deferred
    ``_cfg_ops`` queue.

    PPO clips the policy gradient surrogate objective to prevent destructively
    large policy updates (Schulman et al., 2017):

    .. math::

        L^{\\text{CLIP}}(\\theta) = \\mathbb{E}_t\\!\\left[
            \\min\\!\\left(r_t(\\theta)\\hat{A}_t,\\;
            \\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon)\\hat{A}_t
            \\right)
        \\right]

    where :math:`r_t(\\theta) = \\pi_\\theta(a_t|s_t) / \\pi_{\\theta_{\\text{old}}}(a_t|s_t)`.

    References
    ----------
    Schulman, J. et al. (2017) "Proximal Policy Optimization Algorithms"
    arXiv:1707.06347.

    Attributes
    ----------
    algo_class : type[Algorithm]
        Ray RLlib algorithm class used to build the underlying trainer.  Fixed
        to ``PPO`` for this config subclass.

    Examples
    --------
    >>> cfg = (
    ...     PPOptimizerConfig()
    ...     .environment(env=FisherEnv, train_iters=200)
    ...     .training(lr=3e-4, gamma=0.99, clip_param=0.2)
    ...     .env_runners(num_env_runners=4, num_envs_per_env_runner=8)
    ...     .multi_agent(policies={"fisher_0": ...})
    ... )
    >>> optimizer = cfg.build_optimizer(world=world_handle, world_name="run_01",
    ...                                  reporting=reporter_handle)
    """

    algo_class: Algorithm = PPO
