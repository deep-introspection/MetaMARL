import logging
from collections import defaultdict

import numpy as np
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.world.context import Context, EnvStepContext

logger = logging.getLogger(__name__)


class DummyRegulatorEnv(RegulatorEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_metrics: list[dict[str, float]] = []

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, ctxs: list[Context]) -> list[float]:
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
