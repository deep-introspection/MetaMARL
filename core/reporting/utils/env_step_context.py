from __future__ import annotations

import wandb
from typing import Any, Optional

from core.utils import sanitize_key, to_float, flatten_numeric
from core.world.context import Context, EnvStepContext


def _extract_reward_series(reward: Any) -> dict[str, float]:
    if isinstance(reward, dict):
        return {str(agent_id): float(value) for agent_id, value in reward.items()}
    return {"global": float(reward)}


def _extract_action_series(action: Any) -> dict[str, float]:
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


def _table_to_line_series_arrays(
    table: wandb.Table,
    *,
    x_col: str,
    y_col: str,
    series_col: str,
) -> tuple[list[list[float]], list[list[float]], list[str]]:
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


# --------------------------
# env-step main plotter
# --------------------------


def plot_env_step_context(
    wandb_run,
    *,
    ctx: Context,
    prefix: str = "env",
    obs_keys_skip: Optional[set[str]] = None,
) -> None:
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

    # finalize current env-step logging
    wandb_run.log({}, step=step, commit=True)
