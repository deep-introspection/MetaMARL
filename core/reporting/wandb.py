from __future__ import annotations

from typing import Any, Optional

import ray
import wandb

from core.world.context import Context
from core.reporting.utils.ray_new_api_stack import plot_training_results_new_stack
from core.reporting.utils.env_step_context import plot_env_step_context


# TODO inherits from abstract reporter
@ray.remote
class WandbReporter:
    """
    Ray actor that owns a single W&B run.

    Other actors/processes should never receive the raw wandb.Run object.
    They only send serializable payloads to this actor.
    """

    def __init__(
        self,
        *,
        project: str,
        name: str,
        config: Optional[dict[str, Any]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        self._run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            reinit=True,
            settings=wandb.Settings(**(settings or {})),
        )

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        hidden: bool | None = None,
        summary: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if step_metric is not None:
            kwargs["step_metric"] = step_metric
        if hidden is not None:
            kwargs["hidden"] = hidden
        if summary is not None:
            kwargs["summary"] = summary

        self._run.define_metric(name, **kwargs)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self._run.log(payload, step=step)

    def log_many(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self._run.log(record["payload"], step=record.get("step"))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def plot_ray_result(
        self,
        outer_iter: int,
        training_episode: int,
        results: dict[str, Any],
        prefix: str = "rllib",
        # plotting controls
        max_lines_returns: int = 64,
        max_rows_returns: int = 50_000,
        max_rows_per_learner_metric: int = 50_000,
        include_all_modules_in_learner_plots: bool = False,  # usually False
        skip_learner_plot_keys: Optional[set[str]] = None,
        learner_plot_whitelist: Optional[set[str]] = None,
        # UI spam controls
        log_per_policy_learner_scalars: bool = False,
        learner_scalar_whitelist: Optional[set[str]] = None,
    ) -> None:
        plot_training_results_new_stack(
            wandb_run=self._run,
            outer_iter=outer_iter,
            training_episode=training_episode,
            results=results,
            prefix=prefix,
            max_lines_returns=max_lines_returns,
            max_rows_returns=max_rows_returns,
            max_rows_per_learner_metric=max_rows_per_learner_metric,
            include_all_modules_in_learner_plots=include_all_modules_in_learner_plots,
            skip_learner_plot_keys=skip_learner_plot_keys,
            learner_plot_whitelist=learner_plot_whitelist,
            log_per_policy_learner_scalars=log_per_policy_learner_scalars,
            learner_scalar_whitelist=learner_scalar_whitelist,
        )

    def plot_env_step(
        self,
        *,
        ctx: Context,
        prefix: str = "env",
        obs_keys_skip: Optional[set[str]] = None,
    ) -> None:
        plot_env_step_context(
            wandb_run=self._run, ctx=ctx, prefix=prefix, obs_keys_skip=obs_keys_skip
        )
