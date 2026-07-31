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
        self.K = ecology_cfg.get("K")
        # Denormalized threshold for visualization
        self.raw_sustainability_threshold = (
            self.sustainability_threshold * self.K
        )
        self.trajectories: dict[int, list[dict[str, Any]]] = {}
        self.last_metrics: list[dict[str, float]] = []

        target_status = ecology_cfg.get("aggregation_status", "eval")
        self.aggregation_status = MechanismStatus(target_status)

        # tail averaging
        self.fitness_tail_steps = int(
            ecology_cfg.get("fitness_tail_steps", 50)
        )

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

        per_mech_metrics: list[dict[str, float]] = []
        step_ctxs = [
            ctx
            for ctx in ctxs
            if isinstance(ctx.payload, EnvStepContext)
            and ctx.payload.status == self.aggregation_status
        ]
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

        # --- group by mechanism index and seed ---
        by_run: dict[tuple[int, int | None], list[Context]] = defaultdict(list)

        # deduplicate
        seen_steps: set[tuple[int | None, int | None, int]] = set()


        for ctx in step_ctxs:
            s = ctx.payload
            key = (s.mechanism, s.seed, ctx.step)
            if key in seen_steps:
                continue
            seen_steps.add(key)
            by_run[(s.mechanism, s.seed)].append(ctx)

        # logger.info(
        #     "[Regulator] Grouped step contexts | "
        #     f"num_mechanisms={len(by_index)} | "
        #     f"indices={sorted(by_index.keys())}"
        # )

        # TODO when running parallel eval, async may duplicate runs ! should not statistically change the result
        metrics_by_mechanism: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.trajectories = {}

        # min_len = min(len(v) for v in by_index.values())
        # logger.info(
        #     "[Regulator] Aggregating | mechanisms=%d | min_len=%d | total_steps=%d",
        #     len(by_index),
        #     min_len,
        #     len(step_ctxs),
        # )
        # # --- compute elastic truncation length ---
        # max_idx = max(by_index.keys())
        # fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        # First aggregation level : one metric record per mechanism-seed run
        for (idx, seed), steps in by_run.items():
            # Assume env-runner order == step order
            steps = sorted(steps, key=lambda ctx: ctx.step)
            num_steps = len(steps)

            rewards = np.empty(num_steps, dtype=np.float32)
            fish = np.empty(num_steps, dtype=np.float32)
            fines = np.empty(num_steps, dtype=np.float32)

            trajectory: list[dict[str, Any]] = []

            for i, s in enumerate(steps):
                # reward
                r = s.payload.reward

                # TODO this is the mean however this is not good representation for late learning mechanisms
                rewards[i] = np.mean(list(r.values()))

                # Extract fish from info dict
                info = s.payload.info
                first_info = next(iter(info.values()))
                fish[i] = float(first_info["fish_norm"])
                fish_current = float(first_info["fish"])

                
                # Extract fines from info dict
                # TODO this is the mean however this is not good representation for late learning mechanisms
                agent_penalties = [
                    float(agent_info.get("violation_signal", 0.0))
                    for agent_info in info.values()
                    if isinstance(agent_info, dict)
                ]

                step_fines = (
                    float(np.mean(agent_penalties))
                    if agent_penalties
                    else 0.0
                )
                fines[i] = step_fines

                # Denormalize for trajectory storage (visualization uses raw values)
                trajectory.append(
                    {
                        "episode": 0, #TODO get episode
                        "step": s.step, #TODO verify s.step == i
                        "seed": seed,
                        "fish_population": fish_current, # TODO do we want fish norm or fish current ?
                        "reward": float(rewards[i]),
                        "fines": float(step_fines),
                    }
                )

            tail_steps = min(
                self.fitness_tail_steps,
                num_steps,
            )

            tail_start = num_steps - tail_steps

            tail_rewards = rewards[tail_start:]
            tail_fish = fish[tail_start:]
            tail_fines = fines[tail_start:]

            self.trajectories.setdefault(idx, {})[seed] = trajectory

            sustainability_penalties = np.maximum(
                0.0,
                (self.sustainability_threshold - tail_fish)
                / max(1e-6, self.sustainability_threshold),
            )

            metrics_by_mechanism[idx].append(
                {
                    "seed": seed,
                    "mean_reward": float(tail_rewards.mean()),
                    "reward_std": float(tail_rewards.std()),
                    "collapse_rate": float(
                        (tail_fish < self.sustainability_threshold).mean()
                    ),
                    "sustainability_penalty": float(
                        sustainability_penalties.mean()
                    ),
                    "min_fish": float(tail_fish.min()),
                    "mean_fish": float(tail_fish.mean()),
                    "mean_fines": float(tail_fish.mean()),
                }
            )

        max_idx = max(metrics_by_mechanism)
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        for idx, seed_metrics in metrics_by_mechanism.items():
            mean_reward = float(
                np.mean([m["mean_reward"] for m in seed_metrics])
            )
            reward_std = float(
                np.mean([m["reward_std"] for m in seed_metrics])
            )
            collapse_rate = float(
                np.mean([m["collapse_rate"] for m in seed_metrics])
            )
            sustainability_penalty = float(
                np.mean([m["sustainability_penalty"] for m in seed_metrics])
            )
            min_fish = float(
                np.mean([m["min_fish"] for m in seed_metrics])
            )
            mean_fish = float(
                np.mean([m["mean_fish"] for m in seed_metrics])
            )
            mean_fines = float(
                np.mean([m["mean_fines"] for m in seed_metrics])
            )


            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=mean_reward,
                collapse_rate=collapse_rate,
                sustainability_penalty=sustainability_penalty,
                sustainability_weight=self.sustainability_weight,
                total_fines=mean_fines,
                mean_fish=mean_fish,
                min_fish=min_fish,
            )

            objective = float(fitness_ctx.objective_score)
            fitness[idx] = objective

            self._publish(
                MechanismContext(
                    index=idx,
                    seed=None,
                    env_id=self.env_id,
                    status=MechanismStatus.done,
                    mechanism=None,
                    metrics=fitness_ctx,
                )
            )

            per_mech_metrics.append(
                {
                    "idx": idx,
                    "objective": objective,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "collapse_rate": collapse_rate,
                    "min_fish": min_fish,
                    "mean_fish": mean_fish,
                    "total_fines": mean_fines,
                    "num_seeds": float(len(seed_metrics)),
                }
            )

        objectives = np.asarray(
            [m["objective"] for m in per_mech_metrics],
            dtype=np.float32,
        )
        collapse_rates = np.asarray(
            [m["collapse_rate"] for m in per_mech_metrics],
            dtype=np.float32,
        )

        best_position = int(np.argmax(objectives))
        worst_position = int(np.argmin(objectives))

        best = per_mech_metrics[best_position]
        worst = per_mech_metrics[worst_position]

        logger.info(
            "[Regulator][summary] "
            "mean_obj=%.4f | best_obj=%.4f (θ=%d) | "
            "worst_obj=%.4f (θ=%d) | "
            "collapse(mean=%.3f max=%.3f)",
            float(objectives.mean()),
            best["objective"],
            int(best["idx"]),
            worst["objective"],
            int(worst["idx"]),
            float(collapse_rates.mean()),
            float(collapse_rates.max()),
        )

        self.last_metrics = per_mech_metrics

        return fitness.tolist() 