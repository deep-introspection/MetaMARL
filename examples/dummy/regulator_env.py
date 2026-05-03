import logging
from collections import defaultdict
from typing import Any

import numpy as np
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.world.context import Context, EnvStepContext

logger = logging.getLogger(__name__)


class DummyRegulatorEnv(RegulatorEnv):
    """Minimal outer-loop regulator environment for testing and prototyping.

    Implements the :class:`~core.envs.regulator.RegulatorEnv` interface
    with the simplest possible reward aggregation: the ES fitness for each
    mechanism candidate is the mean reward collected by the inner RL agents
    during the corresponding rollout.  There is no sustainability penalty or
    ecology model, making this suitable for use with toy inner environments
    (e.g., CartPole) where only convergence of the bilevel loop is being
    validated.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`~core.envs.regulator.RegulatorEnv`.

    Attributes
    ----------
    last_metrics : list[dict[str, float]]
        Per-mechanism metrics computed during the most recent call to
        :meth:`aggregate_rewards`.  Useful for debugging and logging.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_metrics: list[dict[str, float]] = []

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        """Return the regulator's observation of the inner environment state.

        The dummy regulator does not maintain its own state, so this always
        returns a constant scalar ``0.0``.

        Parameters
        ----------
        obs : ObsType
            Raw inner-environment observation (ignored).

        Returns
        -------
        ObsType
            Constant ``0.0``.
        """
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, ctxs: list[Context]) -> list[float]:
        """Compute per-mechanism ES fitness from inner-loop step contexts.

        Groups :class:`~core.world.context.EnvStepContext` payloads by
        mechanism index and returns the mean reward over all steps as the
        fitness signal for each candidate mechanism.

        Parameters
        ----------
        ctxs : list[Context]
            Flat list of context objects published by the inner environment
            runners during the current outer-loop iteration.

        Returns
        -------
        list[float]
            Fitness values indexed by mechanism index (``-inf`` for any
            mechanism with no associated steps).  Side effect: updates
            :attr:`last_metrics` with per-mechanism diagnostic information.
        """
        step_ctxs = [ctx for ctx in ctxs if isinstance(ctx.payload, EnvStepContext)]
        if not step_ctxs:
            logger.warning("[DummyRegulatorEnv] No EnvStepContext received.")
            return []

        by_mech: dict[int, list[Context]] = defaultdict(list)
        for ctx in step_ctxs:
            mech_idx = ctx.payload.mechanism
            by_mech[mech_idx].append(ctx)

        max_idx = max(by_mech.keys())
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        metrics: list[dict[str, float]] = []

        for idx, steps in by_mech.items():
            rewards = []
            for ctx in steps:
                r = ctx.payload.reward
                if isinstance(r, dict):
                    rewards.append(float(sum(r.values())))
                else:
                    rewards.append(float(r))

            mean_reward = float(np.mean(rewards)) if rewards else -np.inf
            fitness[idx] = mean_reward
            metrics.append(
                {
                    "idx": idx,
                    "objective": mean_reward,
                    "mean_reward": mean_reward,
                    "num_steps": float(len(rewards)),
                }
            )

        self.last_metrics = metrics
        return fitness.tolist()
