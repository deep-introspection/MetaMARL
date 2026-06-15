from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import plotly.graph_objects as go
import wandb

from core.utils import sanitize_key, to_float
from core.world.context import Context, EnvStepContext, MechanismStatus
from core.reporting.utils.env_step_context import _extract_info_series


_ENV_REDUCED_ITER_TABLES: dict[tuple[int, str], wandb.Table] = {}


@dataclass(frozen=True)
class ReductionSpec:
    name: str
    fn: Optional[Callable[[list[Context]], float]] = None


def build_default_fishery_reduction_specs() -> list[ReductionSpec]:
    return [ReductionSpec(name="auto_env_metrics")]
 

def _global_step(outer_iter: int, train_step: int) -> int:
    return int(outer_iter) * 1_000_000 + int(train_step)


def _status_to_phase(status: Any, *, fallback: str = "train") -> str:
    if status is None:
        return fallback

    value = str(getattr(status, "value", status))

    if value == MechanismStatus.eval.value:
        return "eval"
    if value == MechanismStatus.train.value:
        return "train"

    return value


def _mean_agent_values(values_by_agent: dict[str, float]) -> float | None:
    vals = []

    for value in values_by_agent.values():
        fv = to_float(value)
        if fv is not None:
            vals.append(float(fv))

    if not vals:
        return None

    return float(np.mean(vals))


def _ctx_to_metric_rows(ctx: Context) -> list[dict[str, Any]]:
    # KEEP_METRICS = {
    #     "info/reservoir_stage_m",
    #     "info/streamflow_m3s",
    #     "info/total_usage_m3s",
    #     "info/requested_m3s",
    #     "info/allowed_m3s",
    #     "info/quota_penalty",
    #     "info/underuse_penalty",
    #     "info/total_penalty",
    # }
    # KEEP_METRICS = {
    #     "info/reservoir_stage",
    #     "info/reservoir_level_norm",
    #     "info/streamflow_m3s",
    #     "info/precip_mm_day",
    #     "info/temp_c",
    #     "info/total_usage_m3s",
    #     "info/full_required_m3_day",
    #     "info/deficit_mm_day",
    #     "info/quota_penalty",
    # }
    KEEP_METRICS = {
        "info/fish",
        "info/fish_next",
        "info/fish_norm",
        "info/growth",
        "info/growth_noise",
        "info/harvest",
        "info/desired_harvest",
        "info/intrinsic_utility",
        "info/violation_signal",
        "info/H_attempted",
        "info/H_realized",
        "info/harvest_scale",
        "info/below_target_zone",
        "info/target_shortfall",
        "info/B_msy",
        "info/MSY",
        "info/F_msy",
        "info/quota_violation",
        "info/quota_penalty",
        "info/stock_penalty",
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

    info_by_key = _extract_info_series(payload.info)

    for info_key, values_by_agent in info_by_key.items():
        add_metric(
            f"info/{info_key}",
            _mean_agent_values(values_by_agent),
        )

    return rows


def _build_metric_tables(rows: list[dict[str, Any]]) -> dict[str, wandb.Table]:
    tables: dict[str, wandb.Table] = {}

    for row in rows:
        metric = sanitize_key(row["metric"])

        if metric not in tables:
            tables[metric] = wandb.Table(
                columns=[
                    "env_step",
                    "phase",
                    "mechanism",
                    "seed",
                    "env_id",
                    "value",
                ]
            )

        tables[metric].add_data(
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
    x_col: str,
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

    grouped: dict[tuple[str, str], dict[int, dict[str, list[float]]]] = {}

    for row in table.data:
        x = to_float(row[ix])
        y = to_float(row[iy])

        if x is None or y is None:
            continue

        phase = str(row[iphase])
        mechanism = str(row[imech])
        seed = str(row[iseed])
        step = int(x)

        grouped.setdefault((phase, mechanism), {}).setdefault(step, {}).setdefault(
            seed, []
        ).append(float(y))

    curves: dict[tuple[str, str], dict[str, list[float]]] = {}

    for key, step_to_seed_values in grouped.items():
        xs, means, stds, uppers, lowers = [], [], [], [], []

        for step in sorted(step_to_seed_values.keys()):
            seed_means = [
                float(np.mean(vals))
                for vals in step_to_seed_values[step].values()
                if vals
            ]

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

        curves[key] = {
            "x": xs,
            "mean": means,
            "std": stds,
            "upper": uppers,
            "lower": lowers,
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
    curves = _table_to_phase_mechanism_curves(
        table,
        x_col="env_step",
    )

    if not curves:
        return

    fig = _make_train_eval_figure(
        curves=curves,
        title=f"{metric_name.replace('/', ' / ')}: train vs eval over horizon",
        xaxis_title="env_step",
    )

    wandb_run.log(
        {f"{prefix}/plots/{metric_name}/train_vs_eval": fig},
        step=step,
        commit=False,
    )


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
    )

    if not curves:
        return

    fig = _make_train_eval_figure(
        curves=curves,
        title=f"{metric_name}: train vs eval over training",
        xaxis_title="train_step",
    )

    wandb_run.log(
        {f"{prefix}/plots_over_training/{metric_name}/train_vs_eval": fig},
        step=step,
        commit=False,
    )


def _make_train_eval_figure(
    *,
    curves: dict[tuple[str, str], dict[str, list[float]]],
    title: str,
    xaxis_title: str,
) -> go.Figure:
    fig = go.Figure()

    phase_colors = {
        "train": "rgba(31, 119, 180, 1.0)",
        "eval": "rgba(255, 127, 14, 1.0)",
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

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
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

    return fig


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

    def add(
        phase: str,
        mechanism: str,
        seed: str,
        metric: str,
        value: float | None,
    ) -> None:
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
        for metric_name, values in metric_values.items():
            if not values:
                continue

            # clean_name = metric_name.replace("info_", "")
            clean_name = metric_name.split("/")[-1]
            arr = np.asarray(values, dtype=np.float64)

            # if clean_name == "total_usage_m3s":
            #     # Total extraction across the rollout.
            #     value = float(np.sum(arr))
            #     out_name = "total_usage_m3s_sum"

            # else:
            #     # Typical state / penalty / flow level over the rollout.
            #     value = float(np.mean(arr))
            #     out_name = f"{clean_name}_mean"

            if clean_name in {"harvest", "desired_harvest", "H_attempted", "H_realized"}:
                value = float(np.sum(arr))
                out_name = f"{clean_name}_sum"
            else:
                value = float(np.mean(arr))
                out_name = f"{clean_name}_mean"

            add(
                phase,
                mechanism,
                seed,
                out_name,
                value,
            )
    return out


def plot_env_reduced(
    *,
    wandb_run,
    ctxs: list[Context],
    outer_iter: int,
    training_episode: int,
    reducers: list[ReductionSpec] | None = None,
    prefix: str = "env_reduced",
) -> None:
    if wandb_run is None or not ctxs:
        return

    gs = _global_step(outer_iter, training_episode)

    rows: list[dict[str, Any]] = []

    for ctx in ctxs:
        rows.extend(_ctx_to_metric_rows(ctx))

    if not rows:
        return

    metric_tables = _build_metric_tables(rows)

    # 1. Raw rollout plots over horizon/env_step.
    for metric_name, table in metric_tables.items():
        _log_train_eval_shaded_plot(
            wandb_run=wandb_run,
            prefix=prefix,
            metric_name=metric_name,
            table=table,
            step=gs,
        )

    # 2. Reduced plots over training iterations.
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
                columns=[
                    "train_step",
                    "phase",
                    "mechanism",
                    "seed",
                    "value",
                ]
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

    wandb_run.log({}, step=gs, commit=True)