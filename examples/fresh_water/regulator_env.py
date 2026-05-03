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
from examples.bilevel_fishery.contexts import FitnessContext

logger = logging.getLogger(__name__)


class FisheryRegulatorEnv(RegulatorEnv):
    """Outer-loop regulator environment for fishery (and water) mechanism optimization.

    Implements the :class:`~core.envs.regulator.RegulatorEnv` interface for
    the ES outer loop.  After each inner-loop PPO rollout it:

    1. Groups step-level :class:`~core.world.context.EnvStepContext` payloads
       by mechanism candidate index.
    2. Computes episode-level metrics (mean reward, collapse rate, sustainability
       penalty).
    3. Combines them into a scalar ES fitness via
       :meth:`~examples.fresh_water.contexts.FitnessContext.from_metrics`.
    4. Stores per-mechanism trajectories for downstream visualization.

    Parameters
    ----------
    ecology_cfg : dict[str, Any]
        Ecology / sustainability configuration with optional keys:

        * ``sus_weight`` (float): Weight of sustainability penalty in the
          fitness objective.  Default ``5.0``.
        * ``sus_threshold`` (float): Normalised fish-stock fraction below
          which a timestep counts as a sustainability violation.
          Default ``0.1``.
        * ``max_fish`` (float): Maximum fish stock (for denormalization).
          Default ``2.0``.
        * ``max_algae`` (float): Maximum algae stock (for denormalization).
          Default ``2.0``.
    **kwargs
        Forwarded to :class:`~core.envs.regulator.RegulatorEnv`.

    Attributes
    ----------
    trajectories : dict[int, list[dict[str, Any]]]
        Per-mechanism trajectory records populated after each call to
        :meth:`aggregate_rewards`.  Each record contains ``episode``,
        ``step``, ``fish_population``, ``algae_population``, and
        ``reward`` fields.
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
        self.max_fish = ecology_cfg.get("max_fish", 2.0)
        self.max_algae = ecology_cfg.get("max_algae", 2.0)
        # Denormalized threshold for visualization
        self.raw_sustainability_threshold = (
            self.sustainability_threshold * self.max_fish
        )
        self.trajectories: dict[int, list[dict[str, Any]]] = {}

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        """Return the regulator's observation of the inner environment state.

        The ES outer loop does not use an environment observation to select
        mechanisms (it uses the fitness signal instead), so this always
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
        mechanism index, truncates all mechanism sequences to equal length
        (elastic truncation to the shortest), then computes the ES objective
        as ``mean_reward - sustainability_weight * sustainability_penalty``
        for each candidate.  Also populates :attr:`trajectories` with
        denormalized step data for visualization.

        Steps
        -----
        1. Filter to :class:`~core.world.context.EnvStepContext` payloads.
        2. Group by mechanism index.
        3. Truncate all groups to ``min(len(group))`` steps.
        4. Compute ``mean_reward``, ``collapse_rate``, and
           ``sustainability_penalty`` per mechanism.
        5. Build a :class:`~examples.fresh_water.contexts.FitnessContext` and
           publish a :class:`~core.world.context.MechanismContext`.
        6. Return a fitness list indexed by mechanism index.

        Parameters
        ----------
        ctxs : list[Context]
            Flat list of context objects published by inner environment
            runners during the current outer-loop iteration.

        Returns
        -------
        list[float]
            ES fitness values indexed by mechanism index.  Entries for
            indices with no step data are set to ``-inf``.

        Warns
        -----
        Logs a warning if no :class:`~core.world.context.EnvStepContext`
        objects are found in ``ctxs``.
        """

        per_mech_metrics: list[dict[str, float]] = []
        step_ctxs = [ctx for ctx in ctxs if isinstance(ctx.payload, EnvStepContext)]

        # logger.info(
        #     "[Regulator] aggregate_rewards called | "
        #     f"total_ctxs={len(ctxs)} | "
        #     f"step_ctxs={len(step_ctxs)}"
        # )

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

        # logger.info(
        #     "[Regulator] Grouped step contexts | "
        #     f"num_mechanisms={len(by_index)} | "
        #     f"indices={sorted(by_index.keys())}"
        # )
        min_len = min(len(v) for v in by_index.values())
        logger.info(
            "[Regulator] Aggregating | mechanisms=%d | min_len=%d | total_steps=%d",
            len(by_index),
            min_len,
            len(step_ctxs),
        )

        # --- compute elastic truncation length ---
        min_len = min(len(v) for v in by_index.values())
        max_idx = max(by_index.keys())
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        # --- aggregate per mechanism ---
        self.trajectories = {}

        for idx, steps in by_index.items():
            # Assume env-runner order == step order
            steps = steps[:min_len]

            rewards = np.empty(min_len, dtype=np.float32)
            fish = np.empty(min_len, dtype=np.float32)
            algae = np.empty(min_len, dtype=np.float32)
            trajectory: list[dict[str, Any]] = []

            for i, s in enumerate(steps):
                # reward
                r = s.payload.reward
                rewards[i] = sum(r.values()) if isinstance(r, dict) else float(r)

                # fish/algae stock from observation (normalized in [0, 1])
                obs = s.payload.observation
                if isinstance(obs, dict):
                    first_obs = next(iter(obs.values()))
                    fish[i] = first_obs[0]
                    algae[i] = first_obs[1] if len(first_obs) > 1 else 0.0
                else:
                    fish[i] = obs[0]
                    algae[i] = obs[1] if len(obs) > 1 else 0.0

                # Denormalize for trajectory storage (visualization uses raw values)
                trajectory.append(
                    {
                        "episode": 0,
                        "step": i,
                        "fish_population": float(fish[i] * self.max_fish),
                        "algae_population": float(algae[i] * self.max_algae),
                        "reward": float(rewards[i]),
                    }
                )

            self.trajectories[idx] = trajectory

            mean_reward = rewards.mean()
            reward_std = rewards.std()

            min_fish = fish.min()
            mean_fish = fish.mean()

            collapse_mask = fish < self.sustainability_threshold
            collapse_rate = collapse_mask.mean()

            penalties = np.maximum(
                0.0,
                (self.sustainability_threshold - fish)
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
                    "min_fish": min_fish,
                    "mean_fish": mean_fish,
                }
            )

        objectives = np.array(
            [m["objective"] for m in per_mech_metrics], dtype=np.float32
        )
        collapse_rates = np.array(
            [m["collapse_rate"] for m in per_mech_metrics], dtype=np.float32
        )

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
