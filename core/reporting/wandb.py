from __future__ import annotations

from typing import Any, Optional

import ray
import wandb

from core.world.context import Context
from core.reporting.utils.ray_new_api_stack import plot_training_results_new_stack
from core.reporting.utils.env_step_context import plot_env_step_context
from core.reporting.utils.env_reduced import plot_env_reduced, ReductionSpec


# TODO inherits from abstract reporter
@ray.remote
class WandbReporter:
    """Ray remote actor that exclusively owns a single W&B run.

    Centralises all W&B I/O in one actor so that other distributed actors
    (learners, env runners) never hold a ``wandb.Run`` reference directly.
    Callers send serialisable payloads via remote method calls.

    Pre-defined step metrics on initialisation:

    * ``appo/train_step`` — step metric for the ``appo/*`` namespace;
    * ``env_reduced/train_step`` — step metric for the ``env_reduced/*`` and
      ``env_reduced_scalar/*`` namespaces.

    Parameters
    ----------
    project : str
        W&B project name.
    name : str
        Display name for the run.
    config : dict[str, Any] or None
        Experiment hyperparameters logged to W&B.  Defaults to ``{}``.
    settings : dict[str, Any] or None
        Additional ``wandb.Settings`` keyword arguments (e.g. ``{"mode":
        "offline"}``).  Defaults to ``{}``.
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
        self._run.define_metric("appo/train_step")
        self._run.define_metric("appo/*", step_metric="appo/train_step")

        self._run.define_metric("env_reduced/train_step")
        self._run.define_metric("env_reduced/*", step_metric="env_reduced/train_step")

        self._run.define_metric("env_reduced_scalar/*", step_metric="env_reduced/train_step")

    def define_metric(
        self,
        name: str,
        *,
        step_metric: str | None = None,
        hidden: bool | None = None,
        summary: str | None = None,
    ) -> None:
        """Register a W&B metric definition on the underlying run.

        Thin wrapper around ``wandb.Run.define_metric`` that omits ``None``
        kwargs so callers do not need to build the dict themselves.

        Parameters
        ----------
        name : str
            Metric name or glob pattern (e.g. ``"appo/*"``).
        step_metric : str or None
            Name of the metric to use as the x-axis step for ``name``.
        hidden : bool or None
            If ``True``, the metric is hidden in the W&B UI by default.
        summary : str or None
            Aggregation method for the run summary (e.g. ``"max"``,
            ``"last"``).
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
        """Log a single payload dict to the W&B run.

        Parameters
        ----------
        payload : dict[str, Any]
            Metrics to log.  Values must be W&B-serialisable (scalars,
            ``wandb.Image``, ``wandb.Table``, etc.).
        step : int or None
            Global step value.  If ``None``, W&B uses its internal auto-step.
        """
        self._run.log(payload, step=step)

    def log_many(self, records: list[dict[str, Any]]) -> None:
        """Log multiple payload records in sequence.

        Parameters
        ----------
        records : list[dict[str, Any]]
            Each element must have a ``"payload"`` key (dict of metrics) and
            an optional ``"step"`` key (int).
        """
        for record in records:
            self._run.log(record["payload"], step=record.get("step"))

    def finish(self) -> None:
        """Finalise and close the W&B run.

        Marks the run as completed in W&B and releases the run handle.
        Subsequent calls to ``log`` will fail.  Safe to call multiple times.
        """
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
        """Log RLlib new-API-stack training metrics for a single step.

        Delegates to
        :func:`~core.reporting.utils.ray_new_api_stack.plot_training_results_new_stack`.
        See that function for full parameter documentation.

        Parameters
        ----------
        outer_iter : int
            Current ES generation index.
        training_episode : int
            Current inner-loop training step.
        results : dict[str, Any]
            RLlib training result dict.
        prefix : str
            W&B metric namespace prefix.  Defaults to ``"rllib"``.
        max_lines_returns : int
            Maximum number of per-series lines in the returns chart.
        max_rows_returns : int
            Maximum rows in the returns table.
        max_rows_per_learner_metric : int
            Maximum rows per learner-metric table.
        include_all_modules_in_learner_plots : bool
            Include ``"__all_modules__"`` aggregate in learner plots.
        skip_learner_plot_keys : set[str] or None
            Additional learner metric keys to exclude from plots.
        learner_plot_whitelist : set[str] or None
            Learner metric keys for which to create charts.
        log_per_policy_learner_scalars : bool
            Log per-policy learner scalars (verbose; off by default).
        learner_scalar_whitelist : set[str] or None
            Keys to log when ``log_per_policy_learner_scalars`` is ``True``.
        """
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
        """Log per-step environment observations, actions, rewards, and infos to W&B.

        Delegates to
        :func:`~core.reporting.utils.env_step_context.plot_env_step_context`.

        Parameters
        ----------
        ctx : Context
            Context object containing an
            :class:`~core.world.context.EnvStepContext` payload.
        prefix : str
            W&B metric namespace prefix.  Defaults to ``"env"``.
        obs_keys_skip : set[str] or None
            Observation keys to exclude from logging.
        """
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
        """Compute and log episode-level reduced metrics to W&B.

        Delegates to
        :func:`~core.reporting.utils.env_reduced.plot_env_reduced`.

        Parameters
        ----------
        ctxs : list[Context]
            Environment step contexts for the current training episode.
        outer_iter : int
            Current ES generation index.
        training_episode : int
            Current inner-loop training step.
        reducers : list[ReductionSpec]
            Reduction specifications to apply.
        prefix : str
            W&B metric namespace prefix.  Defaults to ``"env_reduced"``.
        """
        plot_env_reduced(
            wandb_run=self._run,
            ctxs=ctxs,
            outer_iter=outer_iter,
            training_episode=training_episode,
            reducers=reducers,
            prefix=prefix,
        )
