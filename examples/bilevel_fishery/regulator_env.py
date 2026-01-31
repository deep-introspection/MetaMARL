from collections import defaultdict
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ObsType
import ray

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.mechanism.base import Mechanism
from core.world.context import Context, EnvStepContext, MechanismContext, MechanismStatus
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
        self.horizon: int = ecology_cfg.get("horizon")
        self.sustainability_weight = ecology_cfg.get("sus_weight", 5.0)
        self.sustainability_threshold = ecology_cfg.get("sus_threshold", 0.1)

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, ctxs: list[Context]) -> list[float]:
        """
        Compute per-mechanism fitness from step-level EnvStepContexts.

        Semantics:
        - Group contexts by mechanism
        - Segment into episodes of length = horizon
        - Drop incomplete episodes
        - Compute episode-level metrics
        - Aggregate exactly like legacy evaluator
        """

        step_ctxs = [
            ctx for ctx in ctxs
            if isinstance(ctx.payload, EnvStepContext)
        ]

        if not step_ctxs:
            return []

        by_index: dict[int, list[Context]] = defaultdict(list)

        for ctx in step_ctxs:
            s = ctx.payload
            by_index[s.mechanism].append(ctx)

        max_idx = max(by_index)
        fitness = [float("-inf")] * (max_idx + 1)

        for idx, ctx_list in by_index.items():
            ctx_list.sort(key=lambda c: c.step)

            steps = [c.payload for c in ctx_list]

            episodes = [
                steps[i : i + self.horizon]
                for i in range(0, len(steps), self.horizon)
                if len(steps[i : i + self.horizon]) == self.horizon
            ]

            # if not episodes:
            #     continue

            episode_rewards: list[float] = []
            episode_min_fish: list[float] = []

            for ep in episodes:
                rewards = []
                fish_vals = []

                for s in ep:
                    # reward
                    if isinstance(s.reward, dict):
                        rewards.append(sum(float(r) for r in s.reward.values()))
                    else:
                        rewards.append(float(s.reward))

                    # fish stock
                    obs = s.observation
                    if isinstance(obs, dict):
                        fish_vals.append(min(float(o[0]) for o in obs.values()))
                    else:
                        fish_vals.append(float(obs[0]))

                episode_rewards.append(float(np.mean(rewards)))
                episode_min_fish.append(float(min(fish_vals)))

            mean_reward = float(np.mean(episode_rewards))
            collapse_rate = float(
                np.mean([mf < self.sustainability_threshold for mf in episode_min_fish])
            )

            penalties = [
                max(
                    0.0,
                    (self.sustainability_threshold - mf)
                    / max(1e-6, self.sustainability_threshold),
                )
                for mf in episode_min_fish
            ]

            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=mean_reward,
                collapse_rate=collapse_rate,
                sustainability_penalty=float(np.mean(penalties)),
                sustainability_weight=self.sustainability_weight,
            )

            # publish result for this ES candidate
            self._publish(
                MechanismContext(
                    index=idx,
                    env_id=self.env_id,
                    status=MechanismStatus.done,
                    job=None,
                    mechanism=None,  # optional now
                    metrics=fitness_ctx,
                )
            )

            # TODO
            fitness[idx] = fitness_ctx.objective_score

        return fitness
