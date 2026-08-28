"""Unit tests for ``core.reporting.utils.env_reduced``.

``plot_env_reduced`` reduces a batch of ``EnvStepContext`` records into long
and wide tables, derived ratios, a correlation matrix, a distribution
summary, per-metric train-versus-eval shaded plots over the episode horizon,
and per-iteration reduced plots accumulated across training. The tests cover
the row extraction (allowlist, farm-area distribution statistics), the
per-phase/per-mechanism curve reduction, the three station observed-versus-
simulated plots and the full logging pipeline on a fake run.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import wandb
from core.reporting.utils import env_reduced as mod
from core.world.context import Context, ContextSchema, MechanismStatus
from tests.reporting.conftest import make_env_ctx

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def test_default_reduction_specs_and_global_step():
    specs = mod.build_default_fishery_reduction_specs()
    assert [s.name for s in specs] == ["auto_env_metrics"]
    assert specs[0].fn is None
    assert mod._global_step(2, 5) == 2_000_005


def test_status_to_phase():
    assert mod._status_to_phase(None) == "train"
    assert mod._status_to_phase(None, fallback="x") == "x"
    assert mod._status_to_phase(MechanismStatus.eval) == "eval"
    assert mod._status_to_phase("train") == "train"
    assert mod._status_to_phase(MechanismStatus.published) == "published"


def test_mean_agent_values():
    assert mod._mean_agent_values({}) is None
    assert mod._mean_agent_values({"a": "nan?", "b": None}) is None
    assert mod._mean_agent_values({"a": 1.0, "b": np.float32(3.0)}) == 2.0


# ---------------------------------------------------------------------------
# row extraction
# ---------------------------------------------------------------------------


def test_ctx_to_metric_rows_rejects_non_env_payload():
    assert mod._ctx_to_metric_rows(None) == []
    bad = Context(id="x", opt_id="o", step=0, env="e", payload=ContextSchema())
    assert mod._ctx_to_metric_rows(bad) == []


def test_ctx_to_metric_rows_allowlist_and_farm_stats():
    ctx = make_env_ctx(
        step=7,
        status=MechanismStatus.eval,
        mechanism=None,
        seed=None,
        env_id=None,
        info={
            "fisher:0": {"fish": 10.0, "not_kept": 1.0, "farm_area_m2": 100.0},
            "fisher:1": {"fish": 20.0, "not_kept": 2.0, "farm_area_m2": 300.0},
        },
    )
    rows = mod._ctx_to_metric_rows(ctx)
    by_metric = {r["metric"]: r for r in rows}
    assert "info_not_kept" not in by_metric
    assert by_metric["info_fish"]["value"] == 15.0
    assert by_metric["info_fish"]["phase"] == "eval"
    assert by_metric["info_fish"]["mechanism"] == "unknown"
    assert by_metric["info_fish"]["seed"] == "unknown"
    assert by_metric["info_fish"]["env_id"] == "unknown"
    assert by_metric["info_fish"]["env_step"] == 7
    expected_stats = {
        "std",
        "min",
        "p05",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    }
    assert {f"info_farm_area_m2_{s}" for s in expected_stats} <= set(by_metric)
    assert by_metric["info_farm_area_m2_min"]["value"] == 100.0
    assert by_metric["info_farm_area_m2_max"]["value"] == 300.0
    assert by_metric["info_farm_area_m2"]["value"] == 200.0


def test_ctx_to_metric_rows_skips_non_numeric_values():
    ctx = make_env_ctx(info={"fisher:0": {"fish": "abc", "quota_source": 1.0}})
    rows = mod._ctx_to_metric_rows(ctx)
    assert [r["metric"] for r in rows] == ["info_quota_source"]


def test_build_metric_tables():
    rows = mod._ctx_to_metric_rows(make_env_ctx(step=1))
    tables = mod._build_metric_tables(rows)
    assert set(tables) >= {"info_fish", "info_H_realized", "info_farm_area_m2"}
    table = tables["info_fish"]
    assert list(table.columns) == [
        "env_step",
        "phase",
        "mechanism",
        "seed",
        "env_id",
        "value",
    ]
    assert table.data[0] == [1, "train", "0", "1", "0", 10.0]


# ---------------------------------------------------------------------------
# curves and shaded figures
# ---------------------------------------------------------------------------


def _curve_table(rows, x_col="env_step"):
    table = wandb.Table(columns=[x_col, "phase", "mechanism", "seed", "value"])
    for row in rows:
        table.add_data(*row)
    return table


def test_table_to_phase_mechanism_curves():
    assert mod._table_to_phase_mechanism_curves(None, x_col="env_step") == {}
    table = _curve_table(
        [
            [0, "train", "0", "1", 1.0],
            [0, "train", "0", "2", 3.0],
            [1, "train", "0", "1", 5.0],
            [0, "eval", "0", "1", 2.0],
        ]
    )
    curves = mod._table_to_phase_mechanism_curves(table, x_col="env_step")
    train = curves[("train", "0")]
    assert train["x"] == [0.0, 1.0]
    assert train["mean"] == [2.0, 5.0]
    assert train["std"] == [1.0, 0.0]
    assert train["upper"] == [3.0, 5.0]
    assert train["lower"] == [1.0, 5.0]
    assert curves[("eval", "0")]["mean"] == [2.0]


def test_make_train_eval_figure_skips_empty_and_unknown_phase():
    curves = {
        ("train", "0"): {
            "x": [0.0, 1.0],
            "mean": [1.0, 2.0],
            "std": [0, 0],
            "upper": [1.0, 2.0],
            "lower": [1.0, 2.0],
        },
        ("other", "1"): {
            "x": [0.0],
            "mean": [1.0],
            "std": [0],
            "upper": [1.0],
            "lower": [1.0],
        },
        ("eval", "1"): {"x": [], "mean": [], "std": [], "upper": [], "lower": []},
    }
    fig = mod._make_train_eval_figure(curves=curves, title="t", xaxis_title="x")
    assert isinstance(fig, go.Figure)
    names = [tr.name for tr in fig.data]
    assert names == [
        "train m0 ±1 std",
        "train m0 mean",
        "other m1 ±1 std",
        "other m1 mean",
    ]
    assert fig.data[2].fillcolor == "rgba(0, 0, 0, 0.20)"


def test_shaded_plot_loggers_return_on_empty_table(fake_run):
    empty = _curve_table([])
    mod._log_train_eval_shaded_plot(
        wandb_run=fake_run, prefix="p", metric_name="m", table=empty, step=0
    )
    mod._log_iteration_reduced_shaded_plot(
        wandb_run=fake_run,
        prefix="p",
        metric_name="m",
        table=_curve_table([], "train_step"),
        step=0,
    )
    assert fake_run.logs == []


def test_shaded_plot_loggers_log_figures(fake_run):
    table = _curve_table([[0, "train", "0", "1", 1.0], [1, "train", "0", "1", 2.0]])
    mod._log_train_eval_shaded_plot(
        wandb_run=fake_run, prefix="p", metric_name="info/x", table=table, step=3
    )
    iter_table = _curve_table([[0, "eval", "0", "1", 1.0]], "train_step")
    mod._log_iteration_reduced_shaded_plot(
        wandb_run=fake_run, prefix="p", metric_name="x_mean", table=iter_table, step=3
    )
    assert fake_run.logs[0][0].keys() == {"p/plots/info/x/train_vs_eval"}
    assert fake_run.logs[1][0].keys() == {"p/plots_over_training/x_mean/train_vs_eval"}
    assert all(isinstance(list(p.values())[0], go.Figure) for p, _, _ in fake_run.logs)


# ---------------------------------------------------------------------------
# per-iteration reduction
# ---------------------------------------------------------------------------


def _row(metric, value, step=0, phase="train", mechanism="0", seed="1"):
    return {
        "env_step": step,
        "phase": phase,
        "mechanism": mechanism,
        "seed": seed,
        "env_id": "0",
        "metric": metric,
        "value": value,
    }


def test_rows_to_iteration_metric_rows_reductions():
    rows = [
        _row("info_H_realized", 1.0, 0),
        _row("info_H_realized", 3.0, 1),
        _row("info_fish", 10.0, 0),
        _row("info_fish", 4.0, 1),
        _row("info_quota_penalty", 0.5, 0),
        _row("info_quota_penalty", 1.5, 1),
        _row("info_growth", np.nan, 0),
    ]
    out = mod._rows_to_iteration_metric_rows(rows, train_step=42)
    values = {r["metric"]: r["value"] for r in out}
    assert values["H_realized_sum"] == 4.0
    assert values["H_realized_mean"] == 2.0
    assert values["H_realized_max"] == 3.0
    assert values["fish_mean"] == 7.0
    assert values["fish_last"] == 4.0
    assert values["fish_min"] == 4.0
    assert values["fish_max"] == 10.0
    assert values["quota_penalty_mean"] == 1.0
    assert "growth_mean" not in values  # NaN reductions are dropped
    assert all(r["train_step"] == 42 for r in out)


# ---------------------------------------------------------------------------
# station observed-vs-simulated plots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "station"),
    [
        (mod._log_02GA041_observed_vs_simulated, "02GA041"),
        (mod._log_02GA014_observed_vs_simulated, "02GA014"),
        (mod._log_West_Montrose_observed_vs_simulated, "West_Montrose"),
    ],
)
def test_station_plots(fake_run, fn, station):
    # missing observed table -> nothing logged
    fn(wandb_run=fake_run, prefix="p", metric_tables={}, step=0)
    assert fake_run.logs == []

    sim = f"info_{station}_streamflow_m3s"
    obs = f"info_{station}_streamflow_m3s_observed"
    rows = [_row(sim, 1.0, 0), _row(sim, 2.0, 1), _row(obs, 1.5, 0), _row(obs, 2.5, 1)]
    tables = mod._build_metric_tables(rows)
    fn(wandb_run=fake_run, prefix="p", metric_tables=tables, step=9)
    payload, step, commit = fake_run.logs[0]
    assert step == 9 and commit is False
    fig = payload[f"p/plots/{station}_simulated_vs_observed"]
    assert [tr.name for tr in fig.data] == [
        f"{station} simulated",
        f"{station} observed",
    ]
    assert list(fig.data[0].y) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# plot_env_reduced
# ---------------------------------------------------------------------------


def test_plot_env_reduced_early_returns(fake_run):
    ctx = make_env_ctx()
    mod.plot_env_reduced(wandb_run=None, ctxs=[ctx], outer_iter=0, training_episode=0)
    mod.plot_env_reduced(wandb_run=fake_run, ctxs=[], outer_iter=0, training_episode=0)
    # info keys outside the allowlist yield no rows
    empty = make_env_ctx(info={"fisher:0": {"not_kept": 1.0}})
    mod.plot_env_reduced(
        wandb_run=fake_run, ctxs=[empty], outer_iter=0, training_episode=0
    )
    assert fake_run.logs == []


def _water_ctxs(phase, seed, n_steps=3):
    ctxs = []
    for step in range(n_steps):
        info = {
            "farm:0": {
                "requested_m3_day": 2.0 + step,
                "allowed_m3_day": 1.0,
                "streamflow_m3s": 10.0,
                "total_usage_m3s": 0.5 * step,
                "02GA041_streamflow_m3s": 3.0 + step,
                "02GA041_streamflow_m3s_observed": 3.5 + step,
                "fish": 1.0,
            }
        }
        ctxs.append(make_env_ctx(step=step, status=phase, seed=seed, info=info))
    return ctxs


def test_plot_env_reduced_full_pipeline(fake_run):
    ctxs = (
        _water_ctxs(MechanismStatus.train, seed=1)
        + _water_ctxs(MechanismStatus.train, seed=2)
        + _water_ctxs(MechanismStatus.eval, seed=3)
    )
    mod.plot_env_reduced(
        wandb_run=fake_run,
        ctxs=ctxs,
        outer_iter=1,
        training_episode=2,
        reducers=mod.build_default_fishery_reduction_specs(),
        prefix="red",
    )
    gs = 1_000_002
    keys = fake_run.logged_keys()

    for name in (
        "raw_env_steps",
        "raw_env_steps_wide",
        "raw_env_steps_wide_derived",
        "correlation_matrix",
        "distribution_summary",
        "training_metrics",
    ):
        assert f"red/tables/{name}" in keys
        assert isinstance(fake_run.payload_for(f"red/tables/{name}"), wandb.Table)

    derived = fake_run.payload_for("red/tables/raw_env_steps_wide_derived")
    assert "requested_over_allowed" in derived.columns
    assert "usage_over_streamflow" in derived.columns
    assert len(derived.data) == 9  # 3 phases/seeds x 3 steps

    assert "red/plots/02GA041_simulated_vs_observed" in keys
    assert "red/plots/info_fish/train_vs_eval" in keys
    assert "red/plots_over_training/requested_m3_day_sum/train_vs_eval" in keys
    assert "red/plots_over_training/streamflow_m3s_last/train_vs_eval" in keys
    assert "red/plots_over_training/fish_mean/train_vs_eval" in keys

    # every intermediate log is uncommitted at the global step; the last commits
    assert all(step == gs for _, step, _ in fake_run.logs)
    assert all(commit is False for _, _, commit in fake_run.logs[:-1])
    assert fake_run.logs[-1] == ({}, gs, True)

    cache = mod._ENV_REDUCED_ITER_TABLES
    assert (id(fake_run), "fish_mean") in cache
    # train seeds 1 and 2 plus eval seed 3 -> three rows for this iteration
    assert len(cache[(id(fake_run), "fish_mean")].data) == 3


def test_plot_env_reduced_accumulates_iteration_tables(fake_run):
    for episode in range(2):
        mod.plot_env_reduced(
            wandb_run=fake_run,
            ctxs=[make_env_ctx(step=s) for s in range(2)],
            outer_iter=0,
            training_episode=episode,
        )
    table = mod._ENV_REDUCED_ITER_TABLES[(id(fake_run), "fish_mean")]
    assert [row[0] for row in table.data] == [0, 1]
    assert (
        "env_reduced/plots_over_training/H_realized_sum/train_vs_eval"
        in fake_run.logged_keys()
    )
