from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.rllib.utils.typing import AgentID
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.actor import ActorHandle

from core.annotations import override
from core.optimizers.base import Optimizer
from core.world.base import World
from core.reporting.wandb import WandbReporter
from core.utils import to_float

# TODO temporary
from core.adaptors.ray.utils import (
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
)
from core.reporting.utils.env_reduced import (
    ReductionSpec,
    build_default_fishery_reduction_specs
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it
class RayOptimizer(Optimizer):
    """Ray/RLlib-backed optimizer for bilevel fishery regulation experiments.

    Owns a ``PolicyActor`` Ray remote actor that holds the RLlib
    ``Algorithm`` object.  The ``Algorithm`` never leaves the actor; all
    interactions go through ``ray.get()`` calls on the actor handle.  This
    isolation prevents the main process from being blocked by RLlib's
    internal thread pools and enables clean shutdown semantics.

    Metrics are extracted from each RLlib ``ResultDict`` via the utility
    helpers in ``core.adaptors.ray.utils`` and forwarded to the W&B reporter
    actor after every training step.

    Parameters
    ----------
    config : RayOptimizerConfig
        Frozen optimizer configuration produced by
        ``RayOptimizerConfig.build_optimizer``.
    world : ActorHandle[World]
        Ray remote actor handle for the simulation world, used to retrieve
        the latest environment step contexts for per-episode plotting.
    reporting : ActorHandle[WandbReporter]
        Ray remote actor handle for W&B metric reporting.

    Attributes
    ----------
    policy_actor : ActorHandle[PolicyActor]
        Remote Ray actor that owns and runs the RLlib ``Algorithm``.
    eval_episodes : int
        Number of evaluation episodes, derived from
        ``evaluation_duration / rollout_fragment_length``.
    eval_base_seed : int or None
        Base seed for evaluation environments.
    _training_rewards : list[float]
        Per-step mean episode returns accumulated within the current ES round.
    _training_losses : list[float]
        Per-step policy losses accumulated within the current ES round.
    _es_round : int
        Current outer evolution-strategy round index (incremented on
        ``reset()``).
    _env_reducers : list[ReductionSpec]
        Reduction specs used to aggregate environment step contexts for
        per-episode W&B plots.
    """

    def __init__(
        self,
        # algo: Algorithm,
        config: RayOptimizerConfig,
        world: ActorHandle[World],
        reporting: ActorHandle[WandbReporter],
    ):
        super().__init__(config)
        # self.algo = algo
        self.world = world  # TODO replace by envFactory
        self.reporting = reporting
        # self.eval_episodes = config.eval_episodes
        # TODO fallback if rollout_fragment_length not in eval_cfg
        self.eval_episodes = (
            config.rllib_cfg.evaluation_duration
            // config.rllib_cfg.evaluation_config.get("rollout_fragment_length")
        )
        self.eval_base_seed = config.eval_base_seed
        # self.rollout_fragment_length = config.rollout_fragment_length

        from core.adaptors.ray.policy_actor import PolicyActor

        self.policy_actor = PolicyActor.remote(config.rllib_cfg)

        # Track training metrics for plotting
        self._training_rewards: list[float] = []
        self._training_losses: list[float] = []
        self._es_round: int = 0

        # TODO remove this - temporary for testing
        self._env_reducers: list[ReductionSpec] = getattr(config, "env_reducers", None) or []

        if not self._env_reducers:
            self._env_reducers = build_default_fishery_reduction_specs()

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        """Number of parallel environments per env-runner worker.

        Used by the outer bilevel loop to determine how many mechanism
        configurations can be evaluated simultaneously in a single training
        call.

        Returns
        -------
        int
            Value of ``AlgorithmConfig.num_envs_per_env_runner``.
        """
        return self.config.rllib_cfg.num_envs_per_env_runner

    # TODO move to utils
    def _build_agent_policy_map(self) -> dict[AgentID, str]:
        """Build a mapping from agent IDs to their base policy ID.

        Iterates over ``config.agent_specs`` and constructs the canonical
        agent-ID string ``"{agent_type}:{index}"`` for each agent, mapping
        it to the spec's ``policy`` field.

        Returns
        -------
        dict[AgentID, str]
            Mapping from agent ID (e.g. ``"fisher:0"``) to base policy ID
            (e.g. ``"fisher"``).
        """
        agent_to_policy: dict[AgentID, str] = {}

        for agent_type, spec in self.config.agent_specs.items():
            policy_id = spec["policy"]
            count = spec["count"]

            for i in range(count):
                agent_id = f"{agent_type}:{i}"
                agent_to_policy[agent_id] = policy_id

        return agent_to_policy

    def _get_policy_handle(self, policy_id: str):
        """Retrieve a policy handle compatible with both old and new RLlib API stacks.

        Tries the new ``RLModule`` API first (``Algorithm.get_module``); falls
        back to the classic ``Policy`` API (``Algorithm.get_policy``) if the
        module is not available.

        Parameters
        ----------
        policy_id : str
            The policy ID to look up (e.g. ``"fisher_0"``).

        Returns
        -------
        RLModule or Policy
            The policy or module object for the given ID.
        """
        # RLModule API (newer)
        try:
            return self.algo.get_module(policy_id)
        except Exception:
            # Policy API (older / classic)
            return self.algo.get_policy(policy_id)

    @override(Optimizer)
    def run(self) -> None:
        """Execute one RLlib training iteration and report metrics to W&B.

        Calls ``PolicyActor.train`` on the remote actor, extracts episode
        return, environment step counts, and policy loss from the
        ``ResultDict``, and forwards them to the W&B reporter actor.  If
        environment step contexts are available from the world actor, a
        per-episode reduced summary is also plotted.

        All Ray remote calls are blocking (``ray.get``).
        """
        logger.info("[PPO] Training step started")
        result = ray.get(self.policy_actor.train.remote())
        step = int(to_float(result.get("training_iteration")) or 0)

        # TODO make this more dynamic NEW_STACK
        # TODO move this to world
        ray.get(
            self.reporting.plot_ray_result.remote(
                outer_iter=self._es_round,
                training_episode=step,
                results=result,
                prefix="appo",
            )
        )

        # TODO reduced env episode plotting
        if self._env_reducers:
            latest_env_ctxs = ray.get(
                self.world.get_latest_env_step_contexts.remote(opt_id=self.opt_id)
            )

            if latest_env_ctxs:
                ray.get(
                    self.reporting.plot_env_reduced.remote(
                        ctxs=latest_env_ctxs,
                        outer_iter=self._es_round,
                        training_episode=step,
                        reducers=self._env_reducers,
                        prefix="env_reduced",
                    )
                )

        # TODO temporary to be moved to a logger Extract metrics
        ep_return = get_episode_return_mean(result)
        steps_iter, steps_life = get_env_steps(result)

        # Track metrics
        self._training_rewards.append(ep_return)
        policy_loss = get_policy_loss_if_present(result)
        self._training_losses.append(policy_loss)

        logger.info(
            "[PPO] Training step completed | iter=%d | ep_return=%.4f | env_steps_iter=%d | env_steps_lifetime=%d | policy_loss=%s",
            step,
            ep_return,
            steps_iter,
            steps_life,
            f"{policy_loss:.6f}" if np.isfinite(policy_loss) else "NA",
        )

    @override(Optimizer)
    def evaluate(self) -> None:
        """Run one evaluation pass using the RLlib algorithm's built-in evaluator.

        Delegates to ``PolicyActor.evaluate`` on the remote actor.  The
        evaluation uses the evaluation configuration specified in the
        ``AlgorithmConfig`` (environment, number of episodes, seed, etc.).
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
        self._es_round += 1
        ray.get(self.policy_actor.reset.remote())

    @override(Optimizer)
    def stop(self) -> None:
        """Shut down the RLlib algorithm and release its resources.

        Calls ``PolicyActor.stop`` on the remote actor, which in turn calls
        ``Algorithm.stop()``.  This terminates all env-runner and learner
        worker processes associated with the algorithm.
        """
        ray.get(self.policy_actor.stop.remote())

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        """Persist algorithm weights and state to a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str, optional
            Directory path for the checkpoint.  If ``None``, a default path
            determined by RLlib will be used.

        Returns
        -------
        _TrainingResult
            Ray Train checkpoint result containing the checkpoint path and
            associated metadata.

        Notes
        -----
        Not yet implemented.
        """
        # TODO
        pass
