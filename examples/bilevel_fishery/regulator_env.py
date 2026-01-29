from typing import Any, SupportsFloat

import ray
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from examples.bilevel_fishery.contexts import FitnessContext


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
        Combine inner-loop EnvStep rewards into one scalar fitness.

        Strategy:
          - Mean episodic reward
        """
        ctx_registry = ray.get(self.world.get_ctx_registry.remote())

        fitness_ctxs = [
            ctx.payload
            for ctx in ctx_registry.values()
            if isinstance(ctx.payload, FitnessContext)
        ]

        if not fitness_ctxs:
            raise RuntimeError("No FitnessContext published by inner optimizer")

        f = fitness_ctxs[-1]
        return float(f.objective_score)


# @override(BaseEnv)
# def reward(self, reward: SupportsFloat = 0.0) -> SupportsFloat:
#     if self.inner is None:
#         return reward

#     ctx_registry = ray.get(self.world.get_ctx_registry.remote())

#     fitness_ctxs = [
#         ctx.payload
#         for ctx in ctx_registry.values()
#         if ctx.opt_id == self.inner.opt_id
#         and isinstance(ctx.payload, FitnessContext)
#     ]

#     if not fitness_ctxs:
#         raise RuntimeError("No FitnessContext published by inner optimizer")

#     return float(fitness_ctxs[-1].objective_score)
