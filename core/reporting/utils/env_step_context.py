from __future__ import annotations

import wandb
from typing import Any, Optional

from core.utils import sanitize_key, to_float, flatten_numeric
from core.world.context import Context, EnvStepContext


def _extract_reward_series(reward: Any) -> dict[str, float]:
    """Flatten a reward payload into an ``{agent_id: float}`` mapping.

    Parameters
    ----------
    reward : Any
        Either a dict mapping agent IDs to scalar rewards (multi-agent), or a
        single scalar reward (single-agent / global).

    Returns
    -------
    dict[str, float]
        Mapping from agent-ID string (or ``"global"``) to reward value.
    """
    if isinstance(reward, dict):
        return {str(agent_id): float(value) for agent_id, value in reward.items()}
    return {"global": float(reward)}


def _extract_action_series(action: Any) -> dict[str, float]:
    """Flatten action(s) into a flat ``{key: float}`` mapping.

    Multi-dimensional actions are expanded with ``dim_i`` suffixes.  Handles
    both multi-agent (dict) and single-agent (scalar/array) actions.

    Parameters
    ----------
    action : Any
        Either a dict mapping agent IDs to (possibly array-valued) actions, or
        a single scalar/array action.

    Returns
    -------
    dict[str, float]
        Keys are ``"<agent_id>"`` for 1-D actions, ``"<agent_id>/dim_<i>"``
        for multi-dimensional actions, or ``"global"`` / ``"global/dim_<i>"``
        for non-dict inputs.
    """
    out: dict[str, float] = {}

    if isinstance(action, dict):
        for agent_id, value in action.items():
            flat = flatten_numeric(value)

            if len(flat) == 1:
                out[str(agent_id)] = flat[0]
            else:
                for i, v in enumerate(flat):
                    out[f"{agent_id}/dim_{i}"] = v
        return out

    flat = flatten_numeric(action)
    if len(flat) == 1:
        return {"global": flat[0]}

    for i, v in enumerate(flat):
        out[f"global/dim_{i}"] = v
    return out


def _extract_observation_series(
    observation: Any,
    *,
    observation_map: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Flatten multi-agent observations into ``{obs_key: {agent_id: float}}``.

    Supports dict-valued per-agent observations (named keys) and vector
    observations (flattened with optional name mapping via ``observation_map``).

    Parameters
    ----------
    observation : Any
        Dict mapping agent IDs to their observations.  Each agent observation
        may be a dict of named fields or a numeric array/scalar.
    observation_map : list[str] or None
        Optional ordered list of names for the observation vector dimensions.
        Applied when the per-agent observation is a flat array rather than a
        named dict.  If shorter than the observation vector, remaining dims
        fall back to ``"obs_<i>"``.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{obs_key: {agent_id: value}}`` — one outer key per observation
        dimension, one inner key per agent.

    Raises
    ------
    TypeError
        If ``observation`` is not a dict.
    """
    out: dict[str, dict[str, float]] = {}

    if not isinstance(observation, dict):
        raise TypeError(
            "Expected observation to be a dict-like multi-agent observation."
        )

    for agent_id, agent_obs in observation.items():
        agent_id = str(agent_id)

        if isinstance(agent_obs, dict):
            for obs_key, value in agent_obs.items():
                flat = flatten_numeric(value)

                if len(flat) == 1:
                    out.setdefault(str(obs_key), {})[agent_id] = flat[0]
                else:
                    for i, v in enumerate(flat):
                        mapped_name = (
                            observation_map[i]
                            if observation_map is not None and i < len(observation_map)
                            else f"{obs_key}_{i}"
                        )
                        out.setdefault(str(mapped_name), {})[agent_id] = v
            continue

        flat = flatten_numeric(agent_obs)

        if len(flat) == 1:
            mapped_name = (
                observation_map[0]
                if observation_map is not None and len(observation_map) > 0
                else "observation"
            )
            out.setdefault(str(mapped_name), {})[agent_id] = flat[0]
        else:
            for i, v in enumerate(flat):
                mapped_name = (
                    observation_map[i]
                    if observation_map is not None and i < len(observation_map)
                    else f"obs_{i}"
                )
                out.setdefault(str(mapped_name), {})[agent_id] = v

    return out

def _extract_info_series(info: Any) -> dict[str, dict[str, float]]:
    """Flatten multi-agent info dicts into ``{info_key: {agent_id: float}}``.

    Multi-dimensional info values are expanded with ``"<key>_<i>"`` suffixes.

    Parameters
    ----------
    info : Any
        Dict mapping agent IDs to their per-step info dicts.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{info_key: {agent_id: value}}`` — one outer key per info field
        (or field dimension), one inner key per agent.

    Raises
    ------
    TypeError
        If ``info`` is not a dict, or if the per-agent info block is not a dict.
    """
    out: dict[str, dict[str, float]] = {}

    if not isinstance(info, dict):
        raise TypeError("Expected info to be a dict-like multi-agent info.")

    for agent_id, agent_info in info.items():
        agent_id = str(agent_id)

        if not isinstance(agent_info, dict):
            raise TypeError(
                f"Expected info for agent '{agent_id}' to be a dict."
            )

        for info_key, value in agent_info.items():
            flat = flatten_numeric(value)

            if len(flat) == 1:
                out.setdefault(str(info_key), {})[agent_id] = flat[0]
            else:
                for i, v in enumerate(flat):
                    out.setdefault(f"{info_key}_{i}", {})[agent_id] = v

    return out


def _table_to_line_series_arrays(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    """Pivot a ``wandb.Table`` into ``(xs, ys, keys)`` for ``wandb.plot.line_series``.

    Parameters
    ----------
    table : wandb.Table
        Source table.  Returns empty lists if ``None`` or has no data.
    x_col : str
        Column name to use as x-axis.
    y_col : str
        Column name to use as y-axis.
    series_col : str
        Column whose distinct values define separate chart lines.

    Returns
    -------
    xs : list[list[float]]
        x-values per series, sorted ascending.
    ys : list[list[float]]
        Corresponding y-values per series.
    keys : list[str]
        Series labels in sorted order.
    """
    if table is None or table.data is None:
        return [], [], []

    cols = list(table.columns)
    ix = cols.index(x_col)
    iy = cols.index(y_col)
    iser = cols.index(series_col)

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
    wandb_run,
    plot_key: str,
    table: wandb.Table,
    x_col: str,
    y_col: str,
    series_col: str,
    step: int,
    title: str,
) -> None:
    """Log a multi-line chart to W&B from a persistent table.

    No-op when the table contains no plottable data.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        Active W&B run.
    plot_key : str
        Full metric key under which the chart is logged (no prefix appended).
    table : wandb.Table
        Persistent table used to build the line-series chart.
    x_col : str
        Column to use as x-axis.
    y_col : str
        Column to use as y-axis.
    series_col : str
        Column whose distinct values define chart series/lines.
    step : int
        Global step value for ``wandb_run.log``.
    title : str
        Chart title shown in the W&B UI.
    """
    xs, ys, keys = _table_to_line_series_arrays(
        table,
        x_col=x_col,
        y_col=y_col,
        series_col=series_col,
    )

    payload: dict[str, Any] = {}

    if keys and xs and ys:
        payload[plot_key] = wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=title,
            xname=x_col,
        )

    wandb_run.log(payload, step=step, commit=False)


# --------------------------
# env-step plot caches
# --------------------------

# run_key -> reward table
_ENV_REWARD_TABLES: dict[int, wandb.Table] = {}

# run_key -> action table
_ENV_ACTION_TABLES: dict[int, wandb.Table] = {}

# run_key -> obs_key -> table
_ENV_OBS_TABLES: dict[int, dict[str, wandb.Table]] = {}

# run_key -> info_key -> table
_ENV_INFO_TABLES: dict[int, dict[str, wandb.Table]] = {}

# --------------------------
# env-step main plotter
# --------------------------


def plot_env_step_context(
    wandb_run,
    *,
    ctx: Context,
    prefix: str = "env",
    obs_keys_skip: Optional[set[str]] = None,
    infos_keys_skip: Optional[set[str]] = None,
) -> None:
    """Log a single environment step's observations, actions, rewards, and infos to W&B.

    Extracts all data from ``ctx.payload`` (an :class:`~core.world.context.EnvStepContext`)
    and produces:

    * one multi-line reward chart (lines = agent IDs);
    * one multi-line action chart (lines = agent IDs);
    * one multi-line observation chart per observation key (lines = agent IDs);
    * one multi-line info chart per info key (lines = agent IDs).

    Each chart accumulates data in a persistent per-run ``wandb.Table``.

    Parameters
    ----------
    wandb_run : wandb.sdk.wandb_run.Run
        Active W&B run.  No-op if ``None``.
    ctx : Context
        Context object whose ``payload`` must be an
        :class:`~core.world.context.EnvStepContext`; returns silently otherwise.
    prefix : str
        Metric namespace prefix.  Defaults to ``"env"``.
    obs_keys_skip : set[str] or None
        Observation keys to exclude from logging (e.g. redundant or high-
        dimensional entries).
    infos_keys_skip : set[str] or None
        Info keys to exclude from logging.
    """
    if wandb_run is None:
        return
    if ctx is None or not isinstance(ctx.payload, EnvStepContext):
        return

    step = int(ctx.step)
    run_key = id(wandb_run)
    payload = ctx.payload

    # --------------------------
    # mechanism: log as scalar reference only
    # --------------------------
    mech_payload: dict[str, Any] = {}
    if payload.mechanism is not None:
        mech_payload[f"{prefix}/mechanism/index"] = int(payload.mechanism)
    if mech_payload:
        wandb_run.log(mech_payload, step=step, commit=False)

    # --------------------------
    # reward: one plot, lines = agent_id
    # --------------------------
    reward_table = _ENV_REWARD_TABLES.get(run_key)
    if reward_table is None:
        reward_table = wandb.Table(columns=["env_step", "agent_id", "value"])
        _ENV_REWARD_TABLES[run_key] = reward_table

    reward_by_agent = _extract_reward_series(payload.reward)
    touched_reward = False
    for agent_id, value in reward_by_agent.items():
        reward_table.add_data(step, str(agent_id), float(value))
        touched_reward = True

    if touched_reward:
        _log_multiline_table_as_plot(
            wandb_run=wandb_run,
            plot_key=f"{prefix}/plots/reward",
            table=reward_table,
            x_col="env_step",
            y_col="value",
            series_col="agent_id",
            step=step,
            title="Reward by agent",
        )

    # --------------------------
    # action: one plot, lines = agent_id
    # --------------------------
    action_table = _ENV_ACTION_TABLES.get(run_key)
    if action_table is None:
        action_table = wandb.Table(columns=["env_step", "agent_id", "value"])
        _ENV_ACTION_TABLES[run_key] = action_table

    action_by_agent = _extract_action_series(payload.action)
    touched_action = False
    for agent_id, value in action_by_agent.items():
        action_table.add_data(step, str(agent_id), float(value))
        touched_action = True

    if touched_action:
        _log_multiline_table_as_plot(
            wandb_run=wandb_run,
            plot_key=f"{prefix}/plots/action",
            table=action_table,
            x_col="env_step",
            y_col="value",
            series_col="agent_id",
            step=step,
            title="Action by agent",
        )

    # --------------------------
    # observations: one plot per state key, lines = agent_id
    # --------------------------
    omit = obs_keys_skip or set()
    obs_map = getattr(payload, "observation_map", None)
    obs_tables = _ENV_OBS_TABLES.setdefault(run_key, {})
    obs_by_key = _extract_observation_series(
        payload.observation, observation_map=obs_map
    )

    for obs_key, values_by_agent in obs_by_key.items():
        if obs_key in omit:
            continue
        obs_key_clean = sanitize_key(obs_key)
        table = obs_tables.get(obs_key_clean)
        if table is None:
            table = wandb.Table(columns=["env_step", "agent_id", "value"])
            obs_tables[obs_key_clean] = table

        touched_obs = False
        for agent_id, value in values_by_agent.items():
            table.add_data(step, str(agent_id), float(value))
            touched_obs = True

        if touched_obs:
            _log_multiline_table_as_plot(
                wandb_run=wandb_run,
                plot_key=f"{prefix}/plots/observation/{obs_key_clean}",
                table=table,
                x_col="env_step",
                y_col="value",
                series_col="agent_id",
                step=step,
                title=f"Observation: {obs_key}",
            )

    # INFOS
    info_omit = infos_keys_skip or set()
    info_tables = _ENV_INFO_TABLES.setdefault(run_key, {})
    info_by_key = _extract_info_series(payload.info)

    for info_key, values_by_agent in info_by_key.items():
        if info_key in info_omit:
            continue

        info_key_clean = sanitize_key(info_key)
        table = info_tables.get(info_key_clean)
        if table is None:
            table = wandb.Table(columns=["env_step", "agent_id", "value"])
            info_tables[info_key_clean] = table

        touched_info = False
        for agent_id, value in values_by_agent.items():
            table.add_data(step, str(agent_id), float(value))
            touched_info = True

        if touched_info:
            _log_multiline_table_as_plot(
                wandb_run=wandb_run,
                plot_key=f"{prefix}/plots/info/{info_key_clean}",
                table=table,
                x_col="env_step",
                y_col="value",
                series_col="agent_id",
                step=step,
                title=f"Info: {info_key}",
            )

    # finalize current env-step logging
    wandb_run.log({}, step=step, commit=True)
