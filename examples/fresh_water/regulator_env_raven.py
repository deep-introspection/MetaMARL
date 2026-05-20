import logging
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
from examples.fresh_water.contexts import FitnessContext

logger = logging.getLogger(__name__)


class WaterRegulatorRavenEnv(RegulatorEnv):
    """
    Outer-loop environment for water mechanism optimization.

    Mirrors the fishery regulator but adapted to water-level semantics. Keeps the
    same context processing and FitnessContext production so downstream tooling
    (visualization, loader) can remain unchanged.
    """

    def __init__(
        self,
        *,
        ecology_cfg: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        env_cfg = ecology_cfg or {}
        self.sustainability_weight = env_cfg.get("sus_weight", 5.0)
        self.sustainability_threshold = env_cfg.get("sus_threshold", 0.2)
        self.max_water = env_cfg.get("max_water", 100.0)
        # Denormalized threshold for visualization
        self.raw_sustainability_threshold = self.sustainability_threshold * self.max_water
        self.trajectories: dict[int, list[dict[str, Any]]] = {}

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, ctxs: list[Context]) -> list[float]:
        """
        Compute per-mechanism fitness from step-level EnvStepContexts.
        Implementation mirrors `FisheryRegulatorEnv.aggregate_rewards`.
        """
        per_mech_metrics: list[dict[str, float]] = []
        step_ctxs = [ctx for ctx in ctxs if isinstance(ctx.payload, EnvStepContext)]

        if not step_ctxs:
            logger.warning(
                "[Regulator] No EnvStepContext received — inner loop likely produced no steps"
            )
            return []

        # --- group by mechanism index ---
        by_index: dict[int, list[Context]] = defaultdict(list)

        for ctx in step_ctxs:
            s = ctx.payload
            by_index[s.mechanism].append(ctx)

        min_len = min(len(v) for v in by_index.values())
        logger.info(
            "[Regulator] Aggregating | mechanisms=%d | min_len=%d | total_steps=%d",
            len(by_index),
            min_len,
            len(step_ctxs),
        )

        # --- compute elastic truncation length ---
        max_idx = max(by_index.keys())
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        # --- aggregate per mechanism ---
        self.trajectories = {}

        for idx, steps in by_index.items():
            # Assume env-runner order == step order
            steps = steps[:min_len]

            rewards = np.empty(min_len, dtype=np.float32)
            water = np.empty(min_len, dtype=np.float32)
            trajectory: list[dict[str, Any]] = []

            for i, s in enumerate(steps):
                # reward
                r = s.payload.reward
                rewards[i] = sum(r.values()) if isinstance(r, dict) else float(r)

                # water stock from observation (normalized in [0, 1])
                obs = s.payload.observation
                if isinstance(obs, dict):
                    first_obs = next(iter(obs.values()))
                    water[i] = first_obs[0]
                else:
                    water[i] = obs[0]

                # Denormalize for trajectory storage
                trajectory.append({
                    "episode": 0,
                    "step": i,
                    "water_level": float(water[i] * self.max_water),
                    "reward": float(rewards[i]),
                })

            self.trajectories[idx] = trajectory

            mean_reward = rewards.mean()
            reward_std = rewards.std()

            min_water = water.min()
            mean_water = water.mean()

            collapse_mask = water < self.sustainability_threshold
            collapse_rate = collapse_mask.mean()

            penalties = np.maximum(
                0.0,
                (self.sustainability_threshold - water)
                / max(1e-6, self.sustainability_threshold),
            )

            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=float(mean_reward),
                collapse_rate=float(collapse_rate),
                sustainability_penalty=float(penalties.mean()),
                sustainability_weight=self.sustainability_weight,
            )

            self._publish(
                MechanismContext(
                    index=idx,
                    env_id=self.env_id,
                    status=MechanismStatus.done,
                    job=None,
                    mechanism=None,
                    metrics=fitness_ctx,
                )
            )

            fitness[idx] = fitness_ctx.objective_score

            per_mech_metrics.append(
                {
                    "idx": idx,
                    "objective": fitness_ctx.objective_score,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "collapse_rate": collapse_rate,
                    "min_water": min_water,
                    "mean_water": mean_water,
                }
            )

        objectives = np.array([m["objective"] for m in per_mech_metrics], dtype=np.float32)
        collapse_rates = np.array([m["collapse_rate"] for m in per_mech_metrics], dtype=np.float32)

        best_idx = int(np.argmax(objectives))
        worst_idx = int(np.argmin(objectives))

        best = per_mech_metrics[best_idx]
        worst = per_mech_metrics[worst_idx]

        logger.info(
            "[Regulator][summary] "
            "mean_obj=%.4f | best_obj=%.4f (θ=%d) | worst_obj=%.4f (θ=%d) | "
            "collapse(mean=%.3f max=%.3f)",
            objectives.mean(),
            best["objective"],
            best["idx"],
            worst["objective"],
            worst["idx"],
            collapse_rates.mean(),
            collapse_rates.max(),
        )

        return fitness.tolist()
