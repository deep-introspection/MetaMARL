from collections import defaultdict
from typing import Any

import numpy as np
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.world.context import (
    Context,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)
from examples.bilevel_fishery.contexts import FitnessContext

import logging

logger = logging.getLogger(__name__)



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

        step_ctxs = [ctx for ctx in ctxs if isinstance(ctx.payload, EnvStepContext)]

        logger.info(
            "[Regulator] aggregate_rewards called | "
            f"total_ctxs={len(ctxs)} | "
            f"step_ctxs={len(step_ctxs)}"
        )

        if not step_ctxs:
            logger.warning(
                "[Regulator] No EnvStepContext received — "
                "inner loop likely produced no steps"
            )
            return []

        # --- group by mechanism index ---
        by_index: dict[int, list[Context]] = defaultdict(list)

        for ctx in step_ctxs:
            s = ctx.payload
            by_index[s.mechanism].append(ctx)

        logger.info(
            "[Regulator] Grouped step contexts | "
            f"num_mechanisms={len(by_index)} | "
            f"indices={sorted(by_index.keys())}"
        )

        # --- compute elastic truncation length ---
        lengths = {idx: len(ctx_list) for idx, ctx_list in by_index.items()}
        min_len = min(lengths.values())

        logger.info(
            "[Regulator] Elastic aggregation | "
            f"min_len={min_len} | "
            f"lengths={lengths}"
        )

        max_idx = max(by_index)
        fitness = [float("-inf")] * (max_idx + 1)

        # --- aggregate per mechanism ---
        for idx, ctx_list in by_index.items():
            ctx_list.sort(key=lambda c: c.step)

            # truncate uniformly
            steps = [c.payload for c in ctx_list[:min_len]]

            rewards: list[float] = []
            fish_vals: list[float] = []

            for s in steps:
                # --- reward ---
                if isinstance(s.reward, dict):
                    rewards.append(sum(float(r) for r in s.reward.values()))
                else:
                    rewards.append(float(s.reward))

                # --- fish stock ---
                obs = s.observation
                if isinstance(obs, dict):
                    fish_vals.append(min(float(o[0]) for o in obs.values()))
                else:
                    fish_vals.append(float(obs[0]))

            mean_reward = float(np.mean(rewards))

            collapse_rate = float(
                np.mean(
                    np.array(fish_vals) < self.sustainability_threshold
                )
            )

            penalties = [
                max(
                    0.0,
                    (self.sustainability_threshold - mf)
                    / max(1e-6, self.sustainability_threshold),
                )
                for mf in fish_vals
            ]

            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=mean_reward,
                collapse_rate=collapse_rate,
                sustainability_penalty=float(np.mean(penalties)),
                sustainability_weight=self.sustainability_weight,
            )

            # publish result for this mechanism
            self._publish(
                MechanismContext(
                    index=idx,
                    env_id=self.env_id,
                    status=MechanismStatus.done,
                    job=None,
                    mechanism=None,  # optional
                    metrics=fitness_ctx,
                )
            )

            logger.info(
                f"[Regulator] Publishing fitness | "
                f"idx={idx} | "
                f"objective={fitness_ctx.objective_score:.4f} | "
                f"collapse_rate={collapse_rate:.3f}"
            )

            fitness[idx] = fitness_ctx.objective_score

        return fitness
