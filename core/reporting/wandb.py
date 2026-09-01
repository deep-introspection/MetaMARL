"""Weights & Biases reporting actor.

``WandbReporter`` is the only reporting backend of this branch: a Ray actor
that owns one ``wandb.Run`` and turns serializable payloads (RLlib result
dicts, ``EnvStepContext`` records, ES populations) into W&B scalars and
plots through the helpers in ``core.reporting.utils``. The ``World`` and the
optimizers hold its ``ActorHandle`` and call the ``plot_*`` methods remotely;
the run object itself never leaves the actor.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import ray

import wandb
from core.reporting.utils.env_reduced import ReductionSpec, plot_env_reduced
from core.reporting.utils.env_step_context import plot_env_step_context
from core.reporting.utils.es_population import (
    plot_es_population as plot_es_population_util,
)
from core.reporting.utils.ray_new_api_stack import plot_training_results_new_stack
from core.world.context import Context


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
        self._defined_prefixes: set[str] = set()
        self._run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            reinit=True,
            settings=wandb.Settings(**(settings or {})),
        )

    def _ensure_prefix_metrics(self, prefix: str) -> None:
        if prefix in self._defined_prefixes:
            return

        step_key = f"{prefix}/train_step"
        self._run.define_metric(step_key)
        self._run.define_metric(f"{prefix}/*", step_metric=step_key)

        self._defined_prefixes.add(prefix)

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        hidden: bool | None = None,
        summary: str | None = None,
    ) -> None:
        """Forward a ``wandb.Run.define_metric`` call with the given options.

        Only the options that are not ``None`` are passed on, so the W&B
        defaults apply to the others.
        """
        kwargs: dict[str, Any] = {}
        if step_metric is not None:
            kwargs["step_metric"] = step_metric
        if hidden is not None:
            kwargs["hidden"] = hidden
        if summary is not None:
            kwargs["summary"] = summary

        self._run.define_metric(name, **kwargs)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        """Log one dictionary of scalars to the run, optionally at ``step``."""
        self._run.log(payload, step=step)

    def log_many(self, records: list[dict[str, Any]]) -> None:
        """Log several records, each a dict with ``payload`` and an optional ``step``."""
        for record in records:
            self._run.log(record["payload"], step=record.get("step"))

    def finish(self) -> None:
        """Close the W&B run; further calls are no-ops."""
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
        # MODIFIED: glue flags forwarded into plot_training_results_new_stack
        log_per_series_return_scalars: bool = False,
        log_return_multiline_plot: bool = False,
        log_learner_multiline_plots: bool = False,
        log_mechanism_shaded_plots: bool = True,
        log_raw_rllib_episode_metrics: bool = False,
    ) -> None:
        """Plot one RLlib training (or evaluation) iteration under ``prefix``.

        Delegates to ``plot_training_results_new_stack`` after defining the
        ``<prefix>/train_step`` step metric. A prefix ending in ``/eval``
        marks evaluation results and disables the learner multi-line plots.
        The remaining keyword arguments are forwarded as plotting and
        UI-spam controls; note that ``log_raw_rllib_episode_metrics`` is
        forced to ``True`` regardless of the value passed.

        Parameters
        ----------
        outer_iter : int
            Outer (ES) generation the iteration belongs to.
        training_episode : int
            Inner training iteration index, used as the step metric.
        results : dict
            Raw RLlib result dictionary of the iteration.
        prefix : str
            Metric namespace, for example ``"appo/train"``.
        """
        self._ensure_prefix_metrics(prefix)
        is_eval = prefix.endswith("/eval")
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
            log_per_series_return_scalars=log_per_series_return_scalars,
            log_return_multiline_plot=log_return_multiline_plot,
            log_learner_multiline_plots=log_learner_multiline_plots and not is_eval,
            log_mechanism_shaded_plots=log_mechanism_shaded_plots,
            log_raw_rllib_episode_metrics=True,
        )

    def plot_env_step(
        self,
        *,
        ctx: Context,
        prefix: str = "env",
        obs_keys_skip: Optional[set[str]] = None,
    ) -> None:
        """Log the scalars of one ``EnvStepContext`` under ``prefix``.

        ``obs_keys_skip`` names observation entries that are not logged.
        """
        self._ensure_prefix_metrics(prefix)
        plot_env_step_context(
            wandb_run=self._run, ctx=ctx, prefix=prefix, obs_keys_skip=obs_keys_skip
        )

    # TODO specific for the environment
    def plot_env_reduced(
        self,
        *,
        ctxs: list[Context],
        outer_iter: int,
        training_episode: int,
        reducers: list[ReductionSpec],
        prefix: str = "env_reduced",
    ) -> None:
        """Reduce a batch of ``EnvStepContext`` records and plot the summaries.

        Each ``ReductionSpec`` in ``reducers`` selects a field and a
        reduction; the resulting per-episode summaries are logged under
        ``prefix`` against ``training_episode`` (see
        ``core.reporting.utils.env_reduced``).
        """
        self._ensure_prefix_metrics(prefix)
        plot_env_reduced(
            wandb_run=self._run,
            ctxs=ctxs,
            outer_iter=outer_iter,
            training_episode=training_episode,
            reducers=reducers,
            prefix=prefix,
        )

    def plot_es_population(
        self,
        *,
        generation: int,
        population: np.ndarray,
        fitness: np.ndarray,
        parameter_names: list[str],
        mean: np.ndarray | None = None,
        sigma: float | None = None,
        best_fitness_global: float | None = None,
        best_candidate_global: np.ndarray | None = None,
        prefix: str = "es",
    ) -> None:
        """
        Plot one outer-optimizer generation.

        All arguments are serializable and may safely be sent to this Ray actor.
        The raw wandb.Run remains owned exclusively by WandbReporter.
        """
        definition_key = f"es_step::{prefix}"

        if definition_key not in self._defined_prefixes:
            generation_key = f"{prefix}/generation"
            self._run.define_metric(generation_key)
            self._run.define_metric(
                f"{prefix}/*",
                step_metric=generation_key,
            )
            self._defined_prefixes.add(definition_key)

        plot_es_population_util(
            wandb_run=self._run,
            generation=generation,
            population=population,
            fitness=fitness,
            parameter_names=parameter_names,
            mean=mean,
            sigma=sigma,
            best_fitness_global=best_fitness_global,
            best_candidate_global=best_candidate_global,
            prefix=prefix,
        )
