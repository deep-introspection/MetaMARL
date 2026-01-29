from typing import Any, SupportsFloat

import numpy as np
import ray
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.world.context import EnvStepContext


class FisheryRegulatorEnv(RegulatorEnv):
    """
    Outer-loop environment for fishery mechanism optimization.

    Responsibilities:
      - Publish candidate mechanisms
      - Run inner PPO optimizer
      - Collect performance metrics
      - Convert to scalar ES reward
    """

    def __init__(
        self,
        *,
        ecology_cfg: dict[str, Any],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sustainability_weight = ecology_cfg.get("sus_weight", 5.0)
        self.sustainability_threshold = ecology_cfg.get("sus_threshold", 0.1)

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, rewards: list[SupportsFloat]) -> SupportsFloat:
        """
        Combine inner-loop EnvStepContexts (step observations, rewards and actions) into one scalar
        fitness.

        Strategy:
          - Mean episodic reward
        """
        ctx_registry = ray.get(self.world.get_ctx_registry.remote())

        steps = [
            ctx.payload
            for ctx in ctx_registry.values()
            if ctx.opt_id == self.inner.opt_id
            and isinstance(ctx.payload, EnvStepContext)
        ]

        # extract metrics
        rewards = np.array([s.reward for s in steps])
        fish = np.array([s.info["fish"] for s in steps])

        mean_reward = rewards.mean()
        min_fish = fish.min()

        collapse = min_fish < self.sustainability_threshold

        sustainability_penalty = max(
            0.0,
            (self.sustainability_threshold - min_fish)
            / max(1e-6, self.sustainability_threshold),
        )

        fitness = mean_reward - self.sustainability_weight * sustainability_penalty

        return float(fitness)
