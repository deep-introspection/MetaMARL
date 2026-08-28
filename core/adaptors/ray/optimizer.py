"""Inner-loop optimizer backed by an RLlib ``Algorithm`` running in Ray.

``RayOptimizer`` is the ``Optimizer`` node the regulator drives: ``run`` is one
RLlib training iteration, ``evaluate`` one fixed-duration evaluation pass and
``reset`` a rebuild of the policy from its initial weights. The algorithm itself
lives in a ``PolicyActor``; this class only forwards calls and keeps light
bookkeeping (inner iteration counter, per-iteration return and loss) for logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import ray
from ray.train._internal.checkpoint_manager import _TrainingResult

# TODO temporary
from core.adaptors.ray.utils import (
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
)
from core.annotations import override
from core.optimizers.base import Optimizer
from core.reporting.utils.env_reduced import (
    ReductionSpec,
    build_default_fishery_reduction_specs,
)
from core.utils import to_float

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):
    """Optimizer wrapping an RLlib algorithm through a ``PolicyActor``.

    Parameters
    ----------
    config : RayOptimizerConfig
        Frozen configuration whose ``rllib_cfg`` is fully resolved (this is
        what ``RayOptimizerConfig.build_optimizer`` passes). A ``PolicyActor``
        is spawned from it immediately.

    Attributes
    ----------
    policy_actor : ActorHandle[PolicyActor]
        Remote owner of the ``Algorithm``.
    eval_episodes : int
        ``evaluation_duration // evaluation_config["rollout_fragment_length"]``,
        an estimate of episodes per evaluation. Raises ``TypeError`` at
        construction when the evaluation config has no
        ``rollout_fragment_length``.
    _inner_iter : int
        Training iterations since the last ``reset``.
    _es_round : int
        Number of ``reset`` calls so far, i.e. the outer (ES) generation.
    _training_rewards, _training_losses : list of float
        Per-iteration mean return and policy loss since the last ``reset``.
    """

    def __init__(
        self,
        # algo: Algorithm,
        config: RayOptimizerConfig,
    ):
        super().__init__(config)
        # self.algo = algo
        # self.eval_episodes = config.eval_episodes
        # TODO fallback if rollout_fragment_length not in eval_cfg
        self.eval_episodes = (
            config.rllib_cfg.evaluation_duration
            // config.rllib_cfg.evaluation_config.get("rollout_fragment_length")
        )
        # self.rollout_fragment_length = config.rollout_fragment_length

        from core.adaptors.ray.policy_actor import PolicyActor

        self.policy_actor = PolicyActor.remote(config.rllib_cfg)

        # Track training metrics for plotting
        self._training_rewards: list[float] = []
        self._training_losses: list[float] = []
        self._inner_iter: int = 0
        self._es_round: int = 0

        # TODO remove this - temporary for testing
        self._env_reducers: list[ReductionSpec] = (
            getattr(config, "env_reducers", None) or []
        )

        if not self._env_reducers:
            self._env_reducers = build_default_fishery_reduction_specs()

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        """Number of mechanism candidates evaluated in parallel per iteration.

        ``RayOptimizerConfig.debugging`` multiplies
        ``num_envs_per_env_runner`` by the number of training seeds, so the
        original count of mechanisms is recovered as
        ``num_envs_per_env_runner // len(seeds)``. The outer ES optimizer
        reads this to size its population.

        Returns
        -------
        int
            Mechanisms per env runner.

        Raises
        ------
        ZeroDivisionError
            If the config has no training seeds (``debugging`` not called or
            called without a seed).
        """
        num_envs = self.config.rllib_cfg.num_envs_per_env_runner
        num_seeds = len(self.config.seeds)
        num_mechanisms = num_envs // num_seeds
        return num_mechanisms

    @override(Optimizer)
    def run(self) -> None:
        """Run one RLlib training iteration on the policy actor.

        Increments the inner iteration counter, extracts the mean episode
        return, the env step counters and (when present) the policy loss from
        the result, appends them to the tracking lists and logs one summary
        line. RLlib's own lifetime ``training_iteration`` is logged for
        reference only; the per-generation counter is ``_inner_iter``.
        Reporting to W&B is currently disabled (commented out).
        """
        logger.info("[PPO] Training step started")
        result = ray.get(self.policy_actor.train.remote())
        # step = int(to_float(result.get("training_iteration")) or 0)

        # Local inner-loop iteration. This resets for each outer ES round.
        self._inner_iter += 1
        step = self._inner_iter

        # RLlib's own lifetime training counter, retained only for debugging.
        rllib_training_iteration = int(to_float(result.get("training_iteration")) or 0)

        # TODO make this more dynamic NEW_STACK
        # TODO move this to world
        # ray.get(
        #     self.reporting.plot_ray_result.remote(
        #         outer_iter=self._es_round,
        #         training_episode=step,
        #         results=result,
        #         prefix="appo/train",
        #     )
        # )

        # eval_result = result.get("evaluation")
        # if eval_result:
        #     ray.get(
        #         self.reporting.plot_ray_result.remote(
        #         outer_iter=self._es_round,
        #         training_episode=step,
        #         results=eval_result,
        #         prefix="appo/eval",
        #     )
        #     )

        # TODO reduced env episode plotting
        # if self._env_reducers:
        #     latest_env_ctxs = ray.get(
        #         self.world.get_new_env_step_contexts.remote(opt_id=self.opt_id)
        #     )

        #     if latest_env_ctxs:
        #         ray.get(
        #             self.reporting.plot_env_reduced.remote(
        #                 ctxs=latest_env_ctxs,
        #                 outer_iter=self._es_round,
        #                 training_episode=step,
        #                 reducers=self._env_reducers,
        #                 prefix="env_reduced",
        #             )
        #         )

        # TODO temporary to be moved to a logger Extract metrics
        ep_return = get_episode_return_mean(result)
        steps_iter, steps_life = get_env_steps(result)

        # Track metrics
        self._training_rewards.append(ep_return)
        policy_loss = get_policy_loss_if_present(result)
        self._training_losses.append(policy_loss)

        logger.info(
            "[PPO] Training step completed | "
            "outer_iter=%d | inner_iter=%d | rllib_iter_lifetime=%d | "
            "ep_return=%.4f | env_steps_iter=%d | "
            "env_steps_lifetime=%d | policy_loss=%s",
            self._es_round,
            step,
            rllib_training_iteration,
            ep_return,
            steps_iter,
            steps_life,
            f"{policy_loss:.6f}" if np.isfinite(policy_loss) else "NA",
        )

    @override(Optimizer)
    def evaluate(self) -> None:
        """Run one evaluation pass on the policy actor and log start/end.

        The evaluation results themselves are not returned; the environments
        publish their ``EnvStepContext`` records to the World, which is where
        the regulator reads the outcome.
        """
        logger.info("[PPO] Evaluation started")
        ray.get(self.policy_actor.evaluate.remote())
        logger.info("[PPO] Evaluation completed")

    @override(Optimizer)
    def reset(self) -> None:
        """Reset policy weights to random initialization."""
        logger.info("[PPO] Resetting policy weights")
        self._training_rewards = []
        self._training_losses = []
        self._inner_iter = 0
        self._es_round += 1
        ray.get(self.policy_actor.reset.remote())

    @override(Optimizer)
    def stop(self) -> None:
        """Stop the RLlib ``Algorithm`` held by the policy actor."""
        ray.get(self.policy_actor.stop.remote())

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        """Checkpointing stub: does nothing and returns ``None``.

        The ``_TrainingResult`` return annotation describes the intended
        contract, not the current behaviour.
        """
        # TODO
        pass
