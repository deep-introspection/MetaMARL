from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import plotly.graph_objects as go
import wandb

from core.utils import sanitize_key, to_float
from core.world.context import Context, EnvStepContext, MechanismStatus

from core.reporting.utils.env_step_context import (
    _extract_action_series,
    _extract_info_series,
    _extract_observation_series,
    _extract_reward_series,
)

_ENV_REDUCED_ITER_TABLES: dict[tuple[int, str], wandb.Table] = {}

@dataclass(frozen=True)
class ReductionSpec:
    """
    Kept for compatibility with existing optimizer/reporter code.

    The new reducer below automatically extracts all currently tracked env metrics:
      - reward
      - action
      - observation/*
      - info/*

    `name` is optional and only used if you later want to pass custom reducers.
    """

    name: str
    fn: Optional[Callable[[list[Context]], float]] = None


def build_default_fishery_reduction_specs() -> list[ReductionSpec]:
    """
    Kept for compatibility.

    We no longer need to manually enumerate fishery metrics here because
    plot_env_reduced() now extracts all numeric reward/action/observation/info
    values directly from EnvStepContext.
    """
    return [ReductionSpec(name="auto_env_metrics")]


def _global_step(outer_iter: int, train_step: int) -> int:
    return int(outer_iter) * 1_000_000 + int(train_step)


def _status_to_phase(status: Any, *, fallback: str = "train") -> str:
    if status is None:
        return fallback

    value = getattr(status, "value", status)
    value = str(value)

    if value == MechanismStatus.eval.value:
        return "eval"
    if value == MechanismStatus.train.value:
        return "train"

    return value


def _mean_agent_values(values_by_agent: dict[str, float]) -> float | None:
    vals: list[float] = []

    for value in values_by_agent.values():
        fv = to_float(value)
        if fv is not None:
            vals.append(float(fv))

    if not vals:
        return None

    return float(np.mean(vals))


def _ctx_to_metric_rows(ctx: Context) -> list[dict[str, Any]]:
    """
    Converts one EnvStepContext into normalized metric rows.

    Each row is one env-level metric value for:
      phase x mechanism x seed x env_step

    Agent-level quantities are averaged inside each env/seed at each step.
    This makes the shaded region represent variability across seeds, not
    artificially inflated variability across agents.
    """
    KEEP_METRICS = {
        "reward",
        "action",
        "observation/fish_norm",
        "observation/algae_norm",
        "info/harvest",
        "info/H_total",
        "info/violation_signal",
        "info/quota_violation",
        "info/preventive_penalty",
        "info/below_target_zone",
        "info/target_shortfall",
    }

    if ctx is None or not isinstance(ctx.payload, EnvStepContext):
        return []

    payload = ctx.payload

    phase = _status_to_phase(getattr(payload, "status", None))
    mechanism = getattr(payload, "mechanism", None)
    seed = getattr(payload, "seed", None)
    env_id = getattr(payload, "env_id", None)
    env_step = int(ctx.step)

    rows: list[dict[str, Any]] = []

    def add_metric(metric_name: str, value: float | None) -> None:
        if metric_name not in KEEP_METRICS:
            return
        
        if value is None:
            return

        fv = to_float(value)
        if fv is None:
            return

        rows.append(
            {
                "env_step": env_step,
                "phase": phase,
                "mechanism": str(mechanism) if mechanism is not None else "unknown",
                "seed": str(seed) if seed is not None else "unknown",
                "env_id": str(env_id) if env_id is not None else "unknown",
                "metric": sanitize_key(metric_name),
                "value": float(fv),
            }
        )

    reward_by_agent = _extract_reward_series(payload.reward)
    add_metric("reward", _mean_agent_values(reward_by_agent))

    action_by_agent = _extract_action_series(payload.action)
    add_metric("action", _mean_agent_values(action_by_agent))

    obs_map = getattr(payload, "observation_map", None)
    obs_by_key = _extract_observation_series(
        payload.observation,
        observation_map=obs_map,
    )

    for obs_key, values_by_agent in obs_by_key.items():
        add_metric(f"observation/{obs_key}", _mean_agent_values(values_by_agent))

    info_by_key = _extract_info_series(payload.info)

    for info_key, values_by_agent in info_by_key.items():
        add_metric(f"info/{info_key}", _mean_agent_values(values_by_agent))

    return rows


def _build_metric_tables(rows: list[dict[str, Any]]) -> dict[str, wandb.Table]:
    tables: dict[str, wandb.Table] = {}

    for row in rows:
        metric = sanitize_key(row["metric"])
        table = tables.get(metric)

        if table is None:
            table = wandb.Table(
                columns=[
                    "env_step",
                    "phase",
                    "mechanism",
                    "seed",
                    "env_id",
                    "value",
                ]
            )
            tables[metric] = table

        table.add_data(
            int(row["env_step"]),
            str(row["phase"]),
            str(row["mechanism"]),
            str(row["seed"]),
            str(row["env_id"]),
            float(row["value"]),
        )

    return tables


def _table_to_phase_mechanism_curves(
    table: wandb.Table,
    *,
    x_col: str = "env_step",
    y_col: str = "value",
    phase_col: str = "phase",
    mechanism_col: str = "mechanism",
    seed_col: str = "seed",
) -> dict[tuple[str, str], dict[str, list[float]]]:
    if table is None or table.data is None:
        return {}

    cols = list(table.columns)
    ix = cols.index(x_col)
    iy = cols.index(y_col)
    iphase = cols.index(phase_col)
    imech = cols.index(mechanism_col)
    iseed = cols.index(seed_col)

    # phase/mechanism -> env_step -> seed -> values
    grouped: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = {}

    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])

        if x is None or y is None:
            continue

        phase = str(row[iphase])
        mechanism = str(row[imech])
        seed = str(row[iseed])

        key = (phase, mechanism)
        step = int(x)

        grouped.setdefault(key, {}).setdefault(step, {}).setdefault(seed, []).append(
            float(y)
        )

    curves: dict[tuple[str, str], dict[str, list[float]]] = {}

    for key, step_to_seed_values in grouped.items():
        xs: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        uppers: list[float] = []
        lowers: list[float] = []
        ns: list[float] = []

        for step in sorted(step_to_seed_values.keys()):
            # Average within seed first, then compute std across seeds.
            seed_means = []

            for vals in step_to_seed_values[step].values():
                if vals:
                    seed_means.append(float(np.mean(vals)))

            if not seed_means:
                continue

            arr = np.asarray(seed_means, dtype=np.float64)
            mean = float(arr.mean())
            std = float(arr.std())

            xs.append(float(step))
            means.append(mean)
            stds.append(std)
            uppers.append(mean + std)
            lowers.append(mean - std)
            ns.append(float(len(seed_means)))

        curves[key] = {
            "x": xs,
            "mean": means,
            "std": stds,
            "upper": uppers,
            "lower": lowers,
            "n": ns,
        }

    return curves


def _log_train_eval_shaded_plot(
    *,
    wandb_run,
    prefix: str,
    metric_name: str,
    table: wandb.Table,
    step: int,
) -> None:
    curves = _table_to_phase_mechanism_curves(table)

    if not curves:
        return

    fig = go.Figure()

    phase_colors = {
        "train": "rgba(31, 119, 180, 1.0)",  # blue
        "eval": "rgba(255, 127, 14, 1.0)",  # orange
    }

    phase_fill_colors = {
        "train": "rgba(31, 119, 180, 0.25)",
        "eval": "rgba(255, 127, 14, 0.25)",
    }

    dash_styles = ["solid", "dash", "dot", "dashdot"]

    mechanisms = sorted({mechanism for _, mechanism in curves.keys()})
    mechanism_dash = {
        mechanism: dash_styles[i % len(dash_styles)]
        for i, mechanism in enumerate(mechanisms)
    }

    # Draw train first, eval second, so eval is easy to see.
    ordered_keys = sorted(
        curves.keys(),
        key=lambda k: (0 if k[0] == "train" else 1, k[1]),
    )

    for phase, mechanism in ordered_keys:
        curve = curves[(phase, mechanism)]

        xs = curve["x"]
        means = curve["mean"]
        upper = curve["upper"]
        lower = curve["lower"]

        if not xs:
            continue

        color = phase_colors.get(phase, "rgba(0, 0, 0, 1.0)")
        fill_color = phase_fill_colors.get(phase, "rgba(0, 0, 0, 0.20)")
        dash = mechanism_dash.get(mechanism, "solid")

        band_x = xs + xs[::-1]
        band_y = upper + lower[::-1]

        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=fill_color, width=0),
                name=f"{phase} m{mechanism} ±1 std",
                hoverinfo="skip",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line=dict(width=3, color=color, dash=dash),
                marker=dict(size=4, color=color),
                name=f"{phase} m{mechanism} mean",
            )
        )

    metric_title = metric_name.replace("/", " / ")

    fig.update_layout(
        title=f"{metric_title}: train vs eval mean ±1 std across seeds",
        xaxis_title="env_step",
        yaxis_title="value",
        hovermode="x unified",
        template="plotly_white",
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(rangeslider_visible=False)

    wandb_run.log(
        {f"{prefix}/plots/{metric_name}/train_vs_eval_by_mechanism_mean_std": fig},
        step=step,
        commit=False,
    )


def _log_metric_scalar_summaries(
    *,
    wandb_run,
    prefix: str,
    metric_name: str,
    table: wandb.Table,
    step: int,
) -> None:
    if table is None or table.data is None:
        return

    cols = list(table.columns)
    iy = cols.index("value")
    iphase = cols.index("phase")
    imech = cols.index("mechanism")

    grouped: dict[tuple[str, str], list[float]] = {}

    for row in table.data:
        y = to_float(row[iy])

        if y is None:
            continue

        phase = str(row[iphase])
        mechanism = str(row[imech])

        grouped.setdefault((phase, mechanism), []).append(float(y))

    payload: dict[str, Any] = {}

    for (phase, mechanism), vals in grouped.items():
        if not vals:
            continue

        arr = np.asarray(vals, dtype=np.float64)
        clean_metric = sanitize_key(metric_name)
        key_base = f"{prefix}/summary/{phase}/m{mechanism}/{clean_metric}"

        payload[f"{key_base}/mean"] = float(arr.mean())
        payload[f"{key_base}/std"] = float(arr.std())
        payload[f"{key_base}/min"] = float(arr.min())
        payload[f"{key_base}/max"] = float(arr.max())

    if payload:
        wandb_run.log(payload, step=step, commit=False)

def _rows_to_iteration_metric_rows(
    rows: list[dict[str, Any]],
    *,
    train_step: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}

    for row in rows:
        key = (row["phase"], row["mechanism"], row["seed"])
        metric = row["metric"]
        grouped.setdefault(key, {}).setdefault(metric, []).append(float(row["value"]))

    out: list[dict[str, Any]] = []

    def add(phase: str, mechanism: str, seed: str, metric: str, value: float | None):
        if value is None or not np.isfinite(value):
            return
        out.append(
            {
                "train_step": int(train_step),
                "phase": phase,
                "mechanism": mechanism,
                "seed": seed,
                "metric": metric,
                "value": float(value),
            }
        )

    for (phase, mechanism, seed), metric_values in grouped.items():
        fish = metric_values.get("observation_fish_norm", [])
        algae = metric_values.get("observation_algae_norm", [])
        h_total = metric_values.get("info_H_total", [])
        harvest = metric_values.get("info_harvest", [])
        target_shortfall = metric_values.get("info_target_shortfall", [])
        quota_violation = metric_values.get("info_quota_violation", [])
        preventive_penalty = metric_values.get("info_preventive_penalty", [])
        below_target_zone = metric_values.get("info_below_target_zone", [])

        add(phase, mechanism, seed, "total_harvest", float(np.sum(h_total)) if h_total else None)
        add(phase, mechanism, seed, "target_shortfall_severity", float(np.mean(target_shortfall)) if target_shortfall else None)
        add(phase, mechanism, seed, "target_shortfall_rate", float(np.mean(np.asarray(target_shortfall) > 0.0)) if target_shortfall else None)
        add(phase, mechanism, seed, "quota_violation_rate", float(np.mean(np.asarray(quota_violation) > 0.0)) if quota_violation else None)
        add(phase, mechanism, seed, "mean_preventive_penalty", float(np.mean(preventive_penalty)) if preventive_penalty else None)
        add(phase, mechanism, seed, "fish_volatility", float(np.std(fish)) if fish else None)
        add(phase, mechanism, seed, "fish_mean", float(np.mean(fish)) if fish else None)
        add(phase, mechanism, seed, "algae_volatility", float(np.std(algae)) if algae else None)
        add(phase, mechanism, seed, "algae_mean", float(np.mean(algae)) if algae else None)
        add(phase, mechanism, seed, "collapse_occupancy_rate", float(np.mean(below_target_zone)) if below_target_zone else None)
        add(phase, mechanism, seed, "collapse_count", float(np.sum(below_target_zone)) if below_target_zone else None)

    return out


def _log_iteration_reduced_shaded_plot(
    *,
    wandb_run,
    prefix: str,
    metric_name: str,
    table: wandb.Table,
    step: int,
) -> None:
    curves = _table_to_phase_mechanism_curves(
        table,
        x_col="train_step",
        y_col="value",
        phase_col="phase",
        mechanism_col="mechanism",
        seed_col="seed",
    )

    if not curves:
        return

    fig = go.Figure()

    phase_colors = {
        "train": "rgba(31, 119, 180, 1.0)",
        "eval": "rgba(255, 127, 14, 1.0)",
    }
    phase_fill_colors = {
        "train": "rgba(31, 119, 180, 0.25)",
        "eval": "rgba(255, 127, 14, 0.25)",
    }

    for phase, mechanism in sorted(curves.keys()):
        curve = curves[(phase, mechanism)]
        xs = curve["x"]
        means = curve["mean"]
        upper = curve["upper"]
        lower = curve["lower"]

        if not xs:
            continue

        color = phase_colors.get(phase, "rgba(0, 0, 0, 1.0)")
        fill_color = phase_fill_colors.get(phase, "rgba(0, 0, 0, 0.20)")

        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=upper + lower[::-1],
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=fill_color, width=0),
                name=f"{phase} m{mechanism} ±1 std",
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=means,
                mode="lines+markers",
                line=dict(width=3, color=color),
                marker=dict(size=5, color=color),
                name=f"{phase} m{mechanism} mean",
            )
        )

    fig.update_layout(
        title=f"{metric_name}: train vs eval over training",
        xaxis_title="train_step",
        yaxis_title="value",
        hovermode="x unified",
        template="plotly_white",
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=False)

    wandb_run.log(
        {f"{prefix}/plots_over_training/{metric_name}/train_vs_eval_by_mechanism_mean_std": fig},
        step=step,
        commit=False,
    )

def plot_env_reduced(
    *,
    wandb_run,
    ctxs: list[Context],
    outer_iter: int,
    training_episode: int,
    reducers: list[ReductionSpec] | None = None,
    prefix: str = "env_reduced",
) -> None:
    """
    Logs train-vs-eval environment dynamics.

    For every currently tracked EnvStepContext metric:
      - reward
      - action
      - observation/*
      - info/*

    this produces one Plotly plot:
      train: blue mean line + blue shaded ±1 std across seeds
      eval: orange mean line + orange shaded ±1 std across seeds

    X-axis:
      env_step within the rollout/episode

    W&B step:
      global optimizer training step = outer_iter * 1_000_000 + training_episode
    """
    if wandb_run is None:
        return
    if not ctxs:
        return

    gs = _global_step(outer_iter, training_episode)

    rows: list[dict[str, Any]] = []

    for ctx in ctxs:
        rows.extend(_ctx_to_metric_rows(ctx))

    if not rows:
        return

    metric_tables = _build_metric_tables(rows)

    iter_rows = _rows_to_iteration_metric_rows(
        rows,
        train_step=gs,
    )

    for row in iter_rows:
        metric_name = sanitize_key(row["metric"])
        cache_key = (id(wandb_run), metric_name)

        table = _ENV_REDUCED_ITER_TABLES.get(cache_key)
        if table is None:
            table = wandb.Table(
                columns=["train_step", "phase", "mechanism", "seed", "value"]
            )
            _ENV_REDUCED_ITER_TABLES[cache_key] = table

        table.add_data(
            int(row["train_step"]),
            str(row["phase"]),
            str(row["mechanism"]),
            str(row["seed"]),
            float(row["value"]),
        )

        _log_iteration_reduced_shaded_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            metric_name=metric_name,
            table=table,
            step=gs,
        )

    for metric_name, table in metric_tables.items():
        # _log_metric_scalar_summaries(
        #     wandb_run=wandb_run,
        #     prefix=prefix,
        #     metric_name=metric_name,
        #     table=table,
        #     step=gs,
        # )

        _log_train_eval_shaded_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            metric_name=metric_name,
            table=table,
            step=gs,
        )

    wandb_run.log({}, step=gs, commit=True)