"""Outer-loop regulator environment for the bilevel fishery experiment.

The :class:`FisheryRegulatorEnv` is the Gymnasium environment seen by the ES
optimizer (outer loop).  Its single ``step`` call triggers a full inner-loop
APPO training run, collects per-mechanism performance metrics from the step
contexts published by the fishing agents, computes a scalar ES fitness for each
evaluated mechanism candidate, and returns it to the ES optimizer.

The fitness function is:

    fitness = mean_reward - sustainability_weight * sustainability_penalty

where ``sustainability_penalty = mean(max(0, threshold - fish) / threshold)``
and all fish values are normalised in [0, 1].
"""

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
    """Outer-loop environment for fishery mechanism optimisation via ES.

    Wraps the inner APPO training loop as a single Gymnasium environment.
    The ES optimizer calls ``step(mechanism_vector)`` once per generation;
    internally this triggers ``train_iters`` APPO training iterations and
    aggregates the resulting step-level contexts into a per-mechanism
    fitness signal.

    Parameters
    ----------
    ecology_cfg : dict of str → Any
        Sustainability configuration with the following optional keys:

        - ``sus_weight`` (float): weight on the sustainability penalty in
          the fitness objective.  Default ``5.0``.
        - ``sus_threshold`` (float): normalised fish stock below which a
          step is counted as a "collapse".  Default ``0.1``.
        - ``max_fish`` (float): carrying capacity (used for de-normalising
          trajectory values for visualisation).  Default ``2.0``.
        - ``max_algae`` (float): carrying capacity for algae.  Default ``2.0``.
    **kwargs
        Forwarded to :class:`~core.envs.regulator.RegulatorEnv`.

    Attributes
    ----------
    sustainability_weight : float
        Penalty weight used in the fitness computation.
    sustainability_threshold : float
        Normalised collapse threshold.
    trajectories : dict of int → list of dict
        Per-mechanism trajectory data (populated after each call to
        :meth:`aggregate_rewards`), keyed by mechanism index.
    last_metrics : list of dict
        Summary statistics for the most recent set of evaluated mechanisms.
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
        self.last_metrics: list[dict[str, float]] = []

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        """Return a dummy scalar observation (ES does not use observations).

        The ES optimizer maintains its own internal state and does not consume
        environment observations.  This method satisfies the
        :class:`~core.envs.regulator.RegulatorEnv` interface.

        Parameters
        ----------
        obs : ObsType
            Ignored.

        Returns
        -------
        float
            Always ``0.0``.
        """
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, ctxs: list[Context]) -> list[float]:
        """Aggregate inner-loop step contexts into per-mechanism ES fitness values.

        Called once per ES generation after all inner-loop steps for the
        current mechanism population have been collected.

        Algorithm
        ---------
        1. Filter contexts to only :class:`~core.world.context.EnvStepContext`
           payloads.
        2. Group by mechanism index (``ctx.payload.mechanism``).
        3. Elastic truncation: trim all groups to the length of the shortest
           group so all mechanisms are evaluated over the same number of steps.
        4. For each mechanism, compute:
           - ``mean_reward``, ``reward_std``
           - ``collapse_rate`` = fraction of steps with ``fish < threshold``
           - ``sustainability_penalty`` = mean normalised stock shortfall
           - ``mean_fish``, ``min_fish``, ``total_fines``
        5. Build a :class:`~examples.bilevel_fishery.contexts.FitnessContext`
           and publish a :class:`~core.world.context.MechanismContext` for
           downstream consumers.
        6. Return a fitness array indexed by mechanism index (``-inf`` for
           missing indices).

        Parameters
        ----------
        ctxs : list of Context
            All step-level contexts published during the inner-loop run.

        Returns
        -------
        list of float
            Per-mechanism fitness values (length = ``max_mechanism_index + 1``).
            Mechanisms that received no steps get ``-inf``.
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
            fines = np.empty(min_len, dtype=np.float32)
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

                # Extract fines from info dict
                info = s.payload.info
                step_fines = 0.0
                if isinstance(info, dict):
                    for agent_info in info.values():
                        if isinstance(agent_info, dict) and "fine" in agent_info:
                            step_fines += agent_info["fine"]
                fines[i] = step_fines

                # Denormalize for trajectory storage (visualization uses raw values)
                trajectory.append(
                    {
                        "episode": 0,
                        "step": i,
                        "fish_population": float(fish[i] * self.max_fish),
                        "algae_population": float(algae[i] * self.max_algae),
                        "reward": float(rewards[i]),
                        "fines": float(step_fines),
                    }
                )

            self.trajectories[idx] = trajectory

            mean_reward = rewards.mean()
            reward_std = rewards.std()

            min_fish = fish.min()
            mean_fish = fish.mean()
            total_fines = fines.sum()

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
                total_fines=float(total_fines),
                mean_fish=float(mean_fish),
                min_fish=float(min_fish),
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
                    "total_fines": total_fines,
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

        # Store metrics for ES logging
        self.last_metrics = per_mech_metrics

        return fitness.tolist()
