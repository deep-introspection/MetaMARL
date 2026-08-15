import logging
from collections import defaultdict
import csv
from pathlib import Path
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

EPS = 1e-8

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
        self.sustainability_weight = env_cfg.get("sustainability_weight", 5.0)

        # self.max_water = env_cfg.get("max_water", 100.0)
        # Denormalized threshold for visualization
        # self.raw_sustainability_threshold = self.sustainability_threshold * self.max_water
        self.trajectories: dict[int, list[dict[str, Any]]] = {}

        # self.economic_weight = env_cfg.get("economic_weight", 1.0)

        target_status = env_cfg.get("aggregation_status", "eval")
        self.aggregation_status = MechanismStatus(target_status)

        # tail averaging
        self.fitness_tail_steps = int(env_cfg.get("fitness_tail_steps", 0))

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
        step_ctxs = [
            ctx
            for ctx in ctxs
            if isinstance(ctx.payload, EnvStepContext)
            and ctx.payload.status == self.aggregation_status
        ]

        if not step_ctxs:
            logger.warning(
                "[Regulator] No EnvStepContext received — "
                "inner loop likely produced no steps"
            )
            return []
    
        by_run: dict[tuple[int, int], list[Context]] = defaultdict(list)

        # deduplicate
        seen_steps: set[tuple[int | None, int | None, int]] = set()

        for ctx in step_ctxs:
            s = ctx.payload
            key = (s.mechanism, s.seed, ctx.step)
            if key in seen_steps:
                continue
            seen_steps.add(key)
            by_run[(s.mechanism, s.seed)].append(ctx)

        metrics_by_mechanism: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.trajectories = {}


        # First aggregation level : one metric record per mechanism-seed run
        for (idx, seed), steps in by_run.items():
            # Assume env-runner order == step order
            steps = sorted(steps, key=lambda ctx: ctx.step)
            num_steps = len(steps)

            rewards = np.empty(num_steps, dtype=np.float32)
            crop_satisfaction = np.empty(num_steps, dtype=np.float32)

            trajectory: list[dict[str, Any]] = []

            for i, s in enumerate(steps):
                r = s.payload.reward
                rewards[i] = np.mean(list(r.values()))

                info = s.payload.info
                agent_crop_satisfaction = [
                    float(
                        agent_info["crop_satisfaction"]
                    )
                    for agent_info in info.values()
                    if isinstance(agent_info, dict)
                    and "crop_satisfaction" in agent_info
                ]
                crop_satisfaction[i] = float(np.mean(agent_crop_satisfaction))

                trajectory.append(
                    {
                        "episode": 0, #TODO get episode
                        "step": s.step, #TODO verify s.step == i
                        "seed": seed,
                        "crop_satisfaction": crop_satisfaction[i],
                        "reward": float(rewards[i]),
                    }
                )
                


            tail_steps = min(self.fitness_tail_steps, num_steps)
            tail_start = num_steps - tail_steps
            tail_rewards = rewards[tail_start:]
            tail_crop_satisfaction = (crop_satisfaction[tail_start:])

            # TODO implement proper tailing
            tail_deviation = compute_streamflow_deviation_from_step_ctx(
                    steps[-1].payload,
                    streamflow_col="Belwood_Lake (res. inflow) [m3/s]"
                )["streamflow_deviation"]

            metrics_by_mechanism[idx].append(
                {
                    "seed": seed,
                    "mean_reward": float(tail_rewards.mean()),
                    "mean_crop_satisfaction": float(tail_crop_satisfaction.mean()),
                    "mean_streamflow_deviation": float(tail_deviation),
                }
            )
        max_idx = max(metrics_by_mechanism)
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        for idx, seed_metrics in metrics_by_mechanism.items():
            mean_reward = float(
                np.mean([m["mean_reward"] for m in seed_metrics])
            )
            mean_crop_satisfaction = float(
                np.mean([
                    m["mean_crop_satisfaction"]
                    for m in seed_metrics
                ])
            )
            mean_streamflow_deviation = float(
                np.mean([
                    m["mean_streamflow_deviation"]
                    for m in seed_metrics
                ])
            )


            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=mean_reward,
                crop_satisfaction=mean_crop_satisfaction,
                streamflow_deviation=mean_streamflow_deviation,
                sustainability_weight=self.sustainability_weight,
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
                "crop_satisfaction": mean_crop_satisfaction,
                "streamflow_deviation": mean_streamflow_deviation,
                "streamflow_score": fitness_ctx.streamflow_score,
                "num_seeds": float(len(seed_metrics)),
            }
        )

        objectives = np.asarray(
            [m["objective"] for m in per_mech_metrics],
            dtype=np.float32,
        )

        best_position = int(np.argmax(objectives))
        worst_position = int(np.argmin(objectives))

        best = per_mech_metrics[best_position]
        worst = per_mech_metrics[worst_position]

        logger.info(
            "[Regulator][summary] "
            "mean_obj=%.4f | best_obj=%.4f (θ=%d) | "
            "worst_obj=%.4f (θ=%d) | ",
            float(objectives.mean()),
            best["objective"],
            int(best["idx"]),
            worst["objective"],
            int(worst["idx"]),
        )

        self.last_metrics = per_mech_metrics

        return fitness.tolist() 


def read_hydrograph_series(
    output_dir: str,
    column_name: str,
) -> dict[str, float]:
    csv_path = Path(output_dir) / "ohms_canshield_Hydrographs.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing hydrograph file: {csv_path}")

    series = {}

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            row = {k.strip(): v for k, v in row.items()}

            date = (
                row.get("date")
                or row.get("Date")
                or row.get("time")
                or row.get("Time")
            )

            if date is None:
                date = next(iter(row.values()))

            if column_name not in row:
                match = next(
                    (k for k in row if k.strip() == column_name.strip()),
                    None,
                )
                if match is None:
                    raise KeyError(
                        f"Column '{column_name}' not found in {csv_path}. "
                        f"Available columns: {list(row.keys())}"
                    )
                column_name = match

            raw = row[column_name]

            if raw not in ("", "---", None):
                series[str(date)] = float(raw)

    return series


def compute_streamflow_deviation(
    *,
    mechanism_output_dir: str,
    baseline_output_dir: str,
    streamflow_col: str,
) -> float:
    q_m = read_hydrograph_series(
        output_dir=mechanism_output_dir,
        column_name=streamflow_col,
    )

    q_0 = read_hydrograph_series(
        output_dir=baseline_output_dir,
        column_name=streamflow_col,
    )

    common_dates = sorted(set(q_m) & set(q_0))

    if not common_dates:
        return float("nan")

    numerator = sum(abs(q_m[t] - q_0[t]) for t in common_dates)
    denominator = sum(abs(q_0[t]) for t in common_dates)

    return numerator / max(EPS, denominator)

def compute_streamflow_deviation_from_step_ctx(
    payload,
    *,
    streamflow_col: str = "West_Montrose [m3/s]",
) -> dict[str, float]:
    first_agent_info = next(iter(payload.info.values()))

    baseline_output_dir = first_agent_info["baseline_ref"]

    baseline_path = Path(baseline_output_dir)

    prepared_runs_dir = baseline_path.parents[1]
    baseline_run_name = baseline_path.parents[0].name

    mechanism_run_name = baseline_run_name.replace("baseline_", "", 1)

    mechanism_output_dir = (
        prepared_runs_dir
        / mechanism_run_name
        / "3_Model_output"
    )

    deviation = compute_streamflow_deviation(
        mechanism_output_dir=str(mechanism_output_dir),
        baseline_output_dir=baseline_output_dir,
        streamflow_col=streamflow_col,
    )

    return {
        "env_id": payload.env_id,
        "seed": payload.seed,
        "mechanism": payload.mechanism,
        "streamflow_col": streamflow_col,
        "streamflow_deviation": deviation,
    }
