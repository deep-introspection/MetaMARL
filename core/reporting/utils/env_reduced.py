from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any, Dict

import numpy as np
import wandb
from wandb.sdk.wandb_run import Run

from core.utils import sanitize_key, to_float
from core.world.context import Context, EnvStepContext

from core.reporting.utils.env_step_context import (
    _extract_action_series,
    _extract_info_series,
    _extract_observation_series,
    _extract_reward_series,
)

logger = logging.getLogger(__name__)

ReducerFn = Callable[["EpisodeSeries"], Optional[float]]


@dataclass(frozen=True)
class EpisodeSeries:
    """Time-series of per-step environment data aggregated across agents.

    Built from a sequence of :class:`~core.world.context.Context` objects by
    :func:`build_episode_series`.  All arrays share the same length (one entry
    per environment step).

    Attributes
    ----------
    steps : np.ndarray
        Integer step indices in ascending order, shape ``(T,)``.
    observation : dict[str, np.ndarray]
        Mapping from observation key to time-series array, shape ``(T,)``.
        Agent values are averaged across agents at each step.
    info : dict[str, np.ndarray]
        Mapping from info key to time-series array, shape ``(T,)``.
        Agent values are averaged across agents at each step.
    reward : dict[str, np.ndarray]
        Mapping from reward key (typically ``"reward_mean"``) to time-series
        array, shape ``(T,)``.
    action : dict[str, np.ndarray]
        Mapping from action key (typically ``"action_mean"``) to time-series
        array, shape ``(T,)``.
    """

    steps: np.ndarray
    observation: dict[str, np.ndarray]
    info: dict[str, np.ndarray]
    reward: dict[str, np.ndarray]
    action: dict[str, np.ndarray]

    def get(self, source: str, key: str) -> np.ndarray:
        """Return a time-series array from the specified data source.

        Parameters
        ----------
        source : str
            One of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``
            (case-insensitive).
        key : str
            Name of the field within the chosen source.

        Returns
        -------
        np.ndarray
            Time-series values for ``key``, or an empty float64 array if
            ``key`` is absent.

        Raises
        ------
        KeyError
            If ``source`` is not one of the four recognised values.
        """
        source = str(source).lower()
        if source == "obs":
            return self.observation.get(key, np.array([], dtype=np.float64))
        if source == "info":
            return self.info.get(key, np.array([], dtype=np.float64))
        if source == "reward":
            return self.reward.get(key, np.array([], dtype=np.float64))
        if source == "action":
            return self.action.get(key, np.array([], dtype=np.float64))
        raise KeyError(
            f"Unknown source '{source}'. Expected one of: obs, info, reward, action"
        )

    def num_steps(self) -> int:
        """Return the number of environment steps in the series.

        Returns
        -------
        int
            Length of the ``steps`` array.
        """
        return int(self.steps.size)


@dataclass(frozen=True)
class ReductionSpec:
    """Specification for reducing an :class:`EpisodeSeries` to a single scalar.

    Pairs a human-readable title and a W&B metric key with a callable
    ``ReducerFn`` that maps an :class:`EpisodeSeries` to an optional float.

    Attributes
    ----------
    key : str
        W&B metric key suffix (will be sanitised by :func:`~core.utils.sanitize_key`
        before logging).
    title : str
        Human-readable label shown on W&B chart axes.
    fn : ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` that computes the
        reduced scalar.  Returns ``None`` when no data is available.
    series_name : str
        Label for the data series in multi-line W&B plots.  Defaults to
        ``"episode"``.
    """

    key: str
    title: str
    fn: ReducerFn
    series_name: str = "episode"

    def __call__(self, episode: EpisodeSeries) -> Optional[float]:
        """Apply the reduction function to ``episode``.

        Parameters
        ----------
        episode : EpisodeSeries
            Episode data to reduce.

        Returns
        -------
        float or None
            Scalar result, or ``None`` if the required data is absent.
        """
        return self.fn(episode)


# run_key -> metric_key -> table
_ENV_REDUCED_TABLES: dict[int, dict[str, wandb.Table]] = {}


def _global_step(outer_iter: int, train_step: int) -> int:
    """Compute a monotonically increasing global step index.

    Encodes both the outer (ES) iteration and the inner (RL) training step
    into a single integer so that W&B charts from different outer iterations
    never overlap on the x-axis.

    Parameters
    ----------
    outer_iter : int
        Current ES generation index.
    train_step : int
        Current inner-loop training step.

    Returns
    -------
    int
        ``outer_iter * 1_000_000 + train_step``.
    """
    return int(outer_iter) * 1_000_000 + int(train_step)


def _cap_table(table: wandb.Table, max_rows: int) -> wandb.Table:
    """Return a ``wandb.Table`` capped to at most ``max_rows`` tail rows.

    When the table already has fewer rows than the cap it is returned
    unchanged.  Otherwise a new table with the same columns is created
    containing only the most-recent ``max_rows`` rows.

    Parameters
    ----------
    table : wandb.Table
        Source table (may be ``None``; returned as-is if ``None``).
    max_rows : int
        Maximum number of rows to retain (tail-truncation).

    Returns
    -------
    wandb.Table
        Possibly-trimmed table.
    """
    if table is None:
        return table
    data = table.data
    if data is None or len(data) <= max_rows:
        return table

    tail = data[-max_rows:]
    new_t = wandb.Table(columns=table.columns)
    for r in tail:
        new_t.add_data(*r)
    return new_t


def _finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert ``values`` to a flat float64 array retaining only finite entries.

    Parameters
    ----------
    values : sequence of float or np.ndarray
        Input values to filter.

    Returns
    -------
    np.ndarray
        1-D float64 array containing only finite (non-NaN, non-Inf) values.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _aggregate_step_values(values_by_agent: dict[str, float]) -> Optional[float]:
    """Average finite values across agents for a single environment step.

    Parameters
    ----------
    values_by_agent : dict[str, float]
        Mapping from agent ID to scalar value at this step.

    Returns
    -------
    float or None
        Mean of all finite agent values, or ``None`` if no finite values exist.
    """
    vals = []
    for v in values_by_agent.values():
        fv = to_float(v)
        if fv is not None and np.isfinite(fv):
            vals.append(float(fv))
    if not vals:
        return None
    return float(np.mean(vals))


def _rows_to_series(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Convert a list of per-step scalar dicts into key-aligned numpy arrays.

    Missing keys in individual rows are filled with ``np.nan``.

    Parameters
    ----------
    rows : list[dict[str, float]]
        One dict per time step; keys are metric names, values are scalars.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from metric name to a float64 array of length ``len(rows)``.
        Returns an empty dict when ``rows`` is empty.
    """
    if not rows:
        return {}

    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())

    out: dict[str, np.ndarray] = {}
    for key in sorted(all_keys):
        vals = []
        for row in rows:
            v = row.get(key, np.nan)
            vals.append(float(v) if v is not None else np.nan)
        out[key] = np.asarray(vals, dtype=np.float64)

    return out


def _table_to_line_series_arrays(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    """Pivot a ``wandb.Table`` into ``(xs, ys, keys)`` for ``wandb.plot.line_series``.

    Rows with non-convertible x or y values are silently skipped.  Each unique
    value in ``series_col`` becomes one line; points within a series are sorted
    by x.

    Parameters
    ----------
    table : wandb.Table
        Source table.  Returns empty lists if ``None`` or has no data.
    x_col : str
        Column name to use as the x-axis.
    y_col : str
        Column name to use as the y-axis.
    series_col : str
        Column name whose distinct values define separate lines.

    Returns
    -------
    xs : list[list[float]]
        x-values per series, sorted ascending.
    ys : list[list[float]]
        Corresponding y-values per series.
    keys : list[str]
        Series labels in sorted order.

    Raises
    ------
    ValueError
        If any of ``x_col``, ``y_col``, or ``series_col`` is absent from the
        table columns.
    """
    if table is None or table.data is None:
        return [], [], []

    cols = list(table.columns)
    try:
        ix = cols.index(x_col)
        iy = cols.index(y_col)
        iser = cols.index(series_col)
    except ValueError as e:
        raise ValueError(
            f"Table missing required columns. Have={cols}, need={[x_col, series_col, y_col]}"
        ) from e

    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])
        s = row[iser]
        if x is None or y is None:
            continue
        grouped.setdefault(str(s), []).append((float(x), float(y)))

    keys = sorted(grouped.keys())
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    for k in keys:
        pts = sorted(grouped[k], key=lambda t: t[0])
        xs.append([p[0] for p in pts])
        ys.append([p[1] for p in pts])

    return xs, ys, keys


def _log_multiline_table_as_plot(
    *,
    wandb_run: Run,
    prefix: str,
    plot_name: str,
    table: wandb.Table,
    x_col: str,
    y_col: str,
    series_col: str,
    step: int,
    max_rows: int,
    title: str,
) -> None:
    """Log a ``wandb.plot.line_series`` chart built from a persistent table.

    The table is first trimmed to ``max_rows`` tail rows to avoid W&B upload
    limits, then converted to (xs, ys, keys) and logged as a multi-line plot.
    No-op if no data remains after trimming.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        Active W&B run.
    prefix : str
        Metric namespace prefix (e.g. ``"env_reduced"``).
    plot_name : str
        Chart name appended to ``prefix/plots/``.
    table : wandb.Table
        Persistent table accumulating rows over training.
    x_col : str
        Column to use as x-axis.
    y_col : str
        Column to use as y-axis.
    series_col : str
        Column whose distinct values define chart series/lines.
    step : int
        Global step value for ``wandb_run.log``.
    max_rows : int
        Maximum number of table rows to include in the chart.
    title : str
        Chart title shown in the W&B UI.
    """
    table = _cap_table(table, max_rows=max_rows)

    xs, ys, keys = _table_to_line_series_arrays(
        table,
        x_col=x_col,
        y_col=y_col,
        series_col=series_col,
    )

    payload: Dict[str, Any] = {}
    if keys and xs and ys:
        payload[f"{prefix}/plots/{plot_name}"] = wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=title,
            xname=x_col,
        )

    wandb_run.log(payload, step=step, commit=False)


def build_episode_series(ctxs: Sequence[Context]) -> EpisodeSeries:
    """Build an :class:`EpisodeSeries` from a sequence of :class:`Context` objects.

    Filters for contexts whose payload is an :class:`~core.world.context.EnvStepContext`,
    sorts them by step index, and aggregates per-agent observations, infos,
    rewards, and actions into mean time-series arrays.

    Parameters
    ----------
    ctxs : Sequence[Context]
        Unordered collection of context objects, potentially containing
        non-environment contexts (these are ignored).

    Returns
    -------
    EpisodeSeries
        Aggregated episode data ready for reduction via :class:`ReductionSpec`.
    """
    env_ctxs = [
        ctx
        for ctx in ctxs
        if ctx is not None and isinstance(getattr(ctx, "payload", None), EnvStepContext)
    ]
    env_ctxs = sorted(env_ctxs, key=lambda c: int(c.step))

    steps: list[int] = []
    obs_rows: list[dict[str, float]] = []
    info_rows: list[dict[str, float]] = []
    reward_rows: list[dict[str, float]] = []
    action_rows: list[dict[str, float]] = []

    for ctx in env_ctxs:
        payload = ctx.payload
        steps.append(int(ctx.step))
        obs_map = getattr(payload, "observation_map", None)

        obs_extracted = _extract_observation_series(
            payload.observation,
            observation_map=obs_map,
        )
        obs_row: dict[str, float] = {}
        for key, values_by_agent in obs_extracted.items():
            agg = _aggregate_step_values(values_by_agent)
            if agg is not None:
                obs_row[str(key)] = agg
        obs_rows.append(obs_row)

        info_extracted = _extract_info_series(payload.info)
        info_row: dict[str, float] = {}
        for key, values_by_agent in info_extracted.items():
            agg = _aggregate_step_values(values_by_agent)
            if agg is not None:
                info_row[str(key)] = agg
        info_rows.append(info_row)

        reward_extracted = _extract_reward_series(payload.reward)
        reward_row: dict[str, float] = {}
        reward_agg = _aggregate_step_values(reward_extracted)
        if reward_agg is not None:
            reward_row["reward_mean"] = reward_agg
        reward_rows.append(reward_row)

        action_extracted = _extract_action_series(payload.action)
        action_row: dict[str, float] = {}
        action_agg = _aggregate_step_values(action_extracted)
        if action_agg is not None:
            action_row["action_mean"] = action_agg
        action_rows.append(action_row)

    return EpisodeSeries(
        steps=np.asarray(steps, dtype=np.int64),
        observation=_rows_to_series(obs_rows),
        info=_rows_to_series(info_rows),
        reward=_rows_to_series(reward_rows),
        action=_rows_to_series(action_rows),
    )


def make_mean_reducer(source: str, key: str) -> ReducerFn:
    """Create a reducer that returns the mean of a time-series field.

    Parameters
    ----------
    source : str
        Data source; one of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``.
    key : str
        Field name within the source.

    Returns
    -------
    ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` computing the mean of
        all finite values, or ``None`` when no data exists.
    """
    def _fn(ep: EpisodeSeries) -> Optional[float]:
        arr = _finite_array(ep.get(source, key))
        if arr.size == 0:
            return None
        return float(np.mean(arr))
    return _fn


def make_sum_reducer(source: str, key: str) -> ReducerFn:
    """Create a reducer that returns the sum of a time-series field.

    Parameters
    ----------
    source : str
        Data source; one of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``.
    key : str
        Field name within the source.

    Returns
    -------
    ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` computing the sum of
        all finite values, or ``None`` when no data exists.
    """
    def _fn(ep: EpisodeSeries) -> Optional[float]:
        arr = _finite_array(ep.get(source, key))
        if arr.size == 0:
            return None
        return float(np.sum(arr))
    return _fn


def make_std_reducer(source: str, key: str) -> ReducerFn:
    """Create a reducer that returns the standard deviation of a time-series field.

    Parameters
    ----------
    source : str
        Data source; one of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``.
    key : str
        Field name within the source.

    Returns
    -------
    ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` computing the std of
        all finite values, or ``None`` when no data exists.
    """
    def _fn(ep: EpisodeSeries) -> Optional[float]:
        arr = _finite_array(ep.get(source, key))
        if arr.size == 0:
            return None
        return float(np.std(arr))
    return _fn


def make_positive_rate_reducer(source: str, key: str) -> ReducerFn:
    """Create a reducer that returns the fraction of strictly positive values.

    Parameters
    ----------
    source : str
        Data source; one of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``.
    key : str
        Field name within the source.

    Returns
    -------
    ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` computing
        ``mean(values > 0)``, or ``None`` when no data exists.
    """
    def _fn(ep: EpisodeSeries) -> Optional[float]:
        arr = _finite_array(ep.get(source, key))
        if arr.size == 0:
            return None
        return float(np.mean(arr > 0.0))
    return _fn


def make_binary_rate_reducer(
    source: str,
    key: str,
    threshold: float = 0.5,
) -> ReducerFn:
    """Create a reducer that returns the fraction of steps above a threshold.

    Parameters
    ----------
    source : str
        Data source; one of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``.
    key : str
        Field name within the source.
    threshold : float
        Value at or above which a step is counted as "active".  Defaults to
        ``0.5`` (binary indicator mid-point).

    Returns
    -------
    ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` computing
        ``mean(values >= threshold)``, or ``None`` when no data exists.
    """
    def _fn(ep: EpisodeSeries) -> Optional[float]:
        arr = _finite_array(ep.get(source, key))
        if arr.size == 0:
            return None
        return float(np.mean(arr >= threshold))
    return _fn


def make_binary_transition_count_reducer(
    source: str,
    key: str,
    threshold: float = 0.5,
) -> ReducerFn:
    """Create a reducer that counts 0→1 transitions in a binarised time series.

    Binarises the time series using ``threshold`` and counts the number of
    off→on transitions.  Useful for counting collapse events (transitions into
    the ``no_fish_zone``).

    Parameters
    ----------
    source : str
        Data source; one of ``"obs"``, ``"info"``, ``"reward"``, ``"action"``.
    key : str
        Field name within the source.
    threshold : float
        Binarisation threshold; steps ``>= threshold`` are treated as ``1``.
        Defaults to ``0.5``.

    Returns
    -------
    ReducerFn
        Callable ``(EpisodeSeries) -> Optional[float]`` returning the number of
        0→1 transitions, or ``0.0`` when the series has fewer than 2 steps.
    """
    def _fn(ep: EpisodeSeries) -> Optional[float]:
        arr = _finite_array(ep.get(source, key))
        if arr.size <= 1:
            return 0.0
        active = (arr >= threshold).astype(np.int32)
        transitions = np.logical_and(active[1:] == 1, active[:-1] == 0)
        return float(np.sum(transitions))
    return _fn


def build_default_fishery_reduction_specs() -> list[ReductionSpec]:
    """Return the standard set of :class:`ReductionSpec` for the fishery environment.

    Covers sustainability indicators (collapse count/rate, fish mean/volatility,
    algae mean/volatility), economic indicators (total harvest, quota violation
    rate), and regulatory-mechanism indicators (shortfall rate/severity,
    preventive penalty).

    Returns
    -------
    list[ReductionSpec]
        Eleven pre-configured reduction specs suitable for passing directly to
        :func:`plot_env_reduced`.
    """
    return [
        ReductionSpec(
            key="collapse_count",
            title="Collapse count",
            fn=make_binary_transition_count_reducer("obs", "no_fish_zone"),
        ),
        ReductionSpec(
            key="collapse_occupancy_rate",
            title="Collapse occupancy rate",
            fn=make_binary_rate_reducer("obs", "no_fish_zone"),
        ),
        ReductionSpec(
            key="target_shortfall_rate",
            title="Target shortfall rate",
            fn=make_binary_rate_reducer("info", "below_target_zone"),
        ),
        ReductionSpec(
            key="target_shortfall_severity",
            title="Target shortfall severity",
            fn=make_mean_reducer("info", "target_shortfall"),
        ),
        ReductionSpec(
            key="total_harvest",
            title="Total harvest",
            fn=make_sum_reducer("info", "H_total"),
        ),
        ReductionSpec(
            key="quota_violation_rate",
            title="Quota violation rate",
            fn=make_positive_rate_reducer("info", "quota_violation"),
        ),
        ReductionSpec(
            key="mean_preventive_penalty",
            title="Mean preventive penalty",
            fn=make_mean_reducer("info", "preventive_penalty"),
        ),
        ReductionSpec(
            key="fish_mean",
            title="Fish mean",
            fn=make_mean_reducer("obs", "fish_norm"),
        ),
        ReductionSpec(
            key="fish_volatility",
            title="Fish volatility",
            fn=make_std_reducer("obs", "fish_norm"),
        ),
        ReductionSpec(
            key="algae_mean",
            title="Algae mean",
            fn=make_mean_reducer("obs", "algae_norm"),
        ),
        ReductionSpec(
            key="algae_volatility",
            title="Algae volatility",
            fn=make_std_reducer("obs", "algae_norm"),
        ),
    ]


def plot_env_reduced(
    wandb_run: Run,
    *,
    ctxs: Sequence[Context],
    outer_iter: int,
    training_episode: int,
    reducers: Sequence[ReductionSpec],
    prefix: str = "env_reduced",
    max_rows_per_metric: int = 50_000,
) -> None:
    """Compute episode-level reduced metrics and log them to W&B.

    For each :class:`ReductionSpec` in ``reducers``:

    1. Applies the reduction function to the aggregated :class:`EpisodeSeries`.
    2. Logs the scalar value under ``{prefix}/{metric_name}``.
    3. Appends a row to a per-metric persistent ``wandb.Table``.
    4. Emits a ``wandb.plot.line_series`` chart for the metric.

    Also logs debug counters (number of contexts, step range) under
    ``{prefix}/debug/``.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        Active W&B run.  No-op if ``None``.
    ctxs : Sequence[Context]
        Environment step contexts for the current training episode.
    outer_iter : int
        Current ES generation index.
    training_episode : int
        Current inner-loop training step.
    reducers : Sequence[ReductionSpec]
        Reduction specifications to apply.
    prefix : str
        Metric namespace prefix in W&B.  Defaults to ``"env_reduced"``.
    max_rows_per_metric : int
        Maximum number of table rows per metric chart.  Defaults to ``50_000``.
    """
    if wandb_run is None or ctxs is None or reducers is None:
        return
    
    gs = _global_step(outer_iter, training_episode)
    ep = build_episode_series(ctxs)
    run_key = id(wandb_run)

    metrics: Dict[str, Any] = {
        f"{prefix}/outer_iter": outer_iter,
        f"{prefix}/train_step": training_episode,
        f"{prefix}/debug/num_ctxs": len(ctxs),
        f"{prefix}/debug/num_episode_steps": ep.num_steps(),
    }

    if ep.num_steps() > 0:
        metrics[f"{prefix}/debug/step_min"] = int(np.min(ep.steps))
        metrics[f"{prefix}/debug/step_max"] = int(np.max(ep.steps))

    tables = _ENV_REDUCED_TABLES.setdefault(run_key, {})
    touched: set[str] = set()

    for spec in reducers:
        try:
            value = spec(ep)
        except Exception:
            logger.exception("[plot_env_reduced] reducer '%s' failed", spec.key)
            continue

        if value is None or not np.isfinite(value):
            continue

        metric_name = sanitize_key(spec.key)
        metrics[f"{prefix}/{metric_name}"] = float(value)
        metrics[f"{prefix}_scalar/{metric_name}"] = float(value)

        table = tables.get(metric_name)
        if table is None:
            table = wandb.Table(
                columns=["step", "outer_iter", "train_step", "series", "value"]
            )
            tables[metric_name] = table

        table.add_data(
            int(gs),
            int(outer_iter),
            int(training_episode),
            str(spec.series_name),
            float(value),
        )
        touched.add(metric_name)

    metrics = {k: v for k, v in metrics.items() if v is not None}
    wandb_run.log(metrics, step=gs, commit=False)

    spec_by_name = {sanitize_key(spec.key): spec for spec in reducers}

    for metric_name in touched:
        table = tables[metric_name]
        spec = spec_by_name[metric_name]

        _log_multiline_table_as_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            plot_name=metric_name,
            table=table,
            x_col="step",
            y_col="value",
            series_col="series",
            step=gs,
            max_rows=max_rows_per_metric,
            title=spec.title,
        )

    wandb_run.log({}, step=gs, commit=False)