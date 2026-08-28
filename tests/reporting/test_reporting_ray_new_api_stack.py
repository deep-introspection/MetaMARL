"""Unit tests for ``core.reporting.utils.ray_new_api_stack``.

The module reads an RLlib new-API-stack ``results`` dictionary (``env_runners``,
``learners``, ``learner_group``, ``timers``), normalises returns per agent per
step, extracts learner scalars and logs them to Weights & Biases with cached
per-run tables and mechanism-level shaded plots. The tests hand-build such a
``results`` dictionary with seeded module ids (``fisher_policy_m0_s1``) and
exercise the extractors, the table-to-curve helpers and the main entry point
with every plotting flag enabled.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

import wandb
from core.reporting.utils import ray_new_api_stack as mod

pytestmark = pytest.mark.unit

M0_S1 = "fisher_policy_m0_s1"
M0_S2 = "fisher_policy_m0_s2"
M1_S1 = "fisher_policy_m1_s1"


def make_results(*, with_learners: bool = True) -> dict:
    """RLlib-shaped results for three seeded fisher modules and two agents."""
    env_runners = {
        "episode_return_mean": 12.0,
        "episode_return_min": 1.0,
        "episode_return_max": 20.0,
        "episode_len_mean": 10.0,
        "episode_len_min": 10.0,
        "episode_len_max": 10.0,
        "num_episodes": 4,
        "num_episodes_lifetime": 40,
        "num_env_steps_sampled": 100,
        "num_env_steps_sampled_lifetime": 1000,
        "num_env_steps_sampled_lifetime_throughput": {
            "throughput_since_last_reduce": 50.0
        },
        "weights_seq_no": 3,
        "num_agent_steps_sampled": {"fisher:0": 100, "fisher:1": 100},
        "num_agent_steps_sampled_lifetime": {"fisher:0": 1000, "fisher:1": 1000},
        "num_module_steps_sampled": {
            M0_S1: 200,
            M0_S2: 200,
            M1_S1: 200,
            "odd_module": 10,
        },
        "module_episode_returns_mean": {
            M0_S1: 10.0,
            M0_S2: 20.0,
            M1_S1: 30.0,
            "odd_module": 1.0,
        },
        "custom_metrics": {"reservoir_stage_m_mean": 4.0, "total_usage_m3s_max": 2.0},
    }
    results = {
        "env_runners": env_runners,
        "timers": {"training_iteration": 1.5, "sample": 0.5},
        "learner_group": {"actor_manager_num_outstanding_async_reqs": 1},
        "mean_num_training_step_calls_since_last_synch_worker_weights": 2,
    }
    if with_learners:
        learner = {
            "total_loss": 1.0,
            "policy_loss": 0.5,
            "vf_loss": 0.4,
            "entropy": 0.8,
            "curr_entropy_coeff": 0.01,
            "diff_num_grad_updates_vs_sampler_policy": 3,
            "num_module_steps_trained": 200,
            "nested": {"ignored": 1},
            "listed": [1, 2],
            "nan_metric": float("nan"),
            "num_module_steps_trained_lifetime_throughput": {
                "throughput_since_last_reduce": 10.0,
                "throughput_since_last_restore": 11.0,
            },
        }
        results["learners"] = {
            M0_S1: dict(learner),
            M0_S2: dict(learner, total_loss=2.0),
            M1_S1: dict(learner, total_loss=3.0),
            "__all_modules__": {
                "learner_thread_in_queue_wait_timer": 0.5,
                "total_loss": 9.0,
            },
            "broken": "not a dict",
        }
    return results


def _table(rows, columns=("step", "outer_iter", "train_step", "series", "value")):
    table = wandb.Table(columns=list(columns))
    for row in rows:
        table.add_data(*row)
    return table


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def test_cap_table():
    assert mod._cap_table(None, 3) is None
    small = _table([[0, 0, 0, "a", 1.0]])
    assert mod._cap_table(small, 3) is small
    big = _table([[i, 0, i, "a", float(i)] for i in range(5)])
    capped = mod._cap_table(big, 2)
    assert [r[0] for r in capped.data] == [3, 4]
    assert list(capped.columns) == list(big.columns)


def test_summarize_dict_of_scalars():
    assert mod._summarize_dict_of_scalars({}) == {}
    assert mod._summarize_dict_of_scalars({"a": float("nan")}) == {}
    out = mod._summarize_dict_of_scalars({"a": 1.0, "b": 3.0, "c": None})
    assert out == {"mean": 2.0, "min": 1.0, "max": 3.0, "std": 1.0, "n": 2.0}


def test_global_step_and_mechanism_id():
    assert mod._global_step(3, 7) == 3_000_007
    assert mod._extract_mechanism_id(M0_S1) == "m0"
    assert mod._extract_mechanism_id("fisher_policy_m12") == "m12"
    assert mod._extract_mechanism_id("m3_s9") == "m3"
    assert mod._extract_mechanism_id("fisher_policy") == "fisher_policy"
    assert mod._extract_mechanism_id("mechanism_0") == "mechanism_0"


def test_should_plot_metric():
    assert mod._should_plot_metric("Total_Loss", {"total_loss"})
    assert not mod._should_plot_metric("lr", {"total_loss"})


# ---------------------------------------------------------------------------
# table -> curves
# ---------------------------------------------------------------------------


def test_table_to_mechanism_mean_std_arrays():
    assert (
        mod._table_to_mechanism_mean_std_arrays(
            None, x_col="step", y_col="value", series_col="series"
        )
        == {}
    )
    with pytest.raises(ValueError, match="missing required columns"):
        mod._table_to_mechanism_mean_std_arrays(
            _table([]), x_col="nope", y_col="value", series_col="series"
        )
    table = _table(
        [
            [0, 0, 0, M0_S1, 1.0],
            [0, 0, 0, M0_S2, 3.0],
            [1, 0, 1, M0_S1, 5.0],
            [0, 0, 0, M1_S1, 7.0],
            [None, 0, 0, M1_S1, 7.0],
        ]
    )
    out = mod._table_to_mechanism_mean_std_arrays(
        table, x_col="step", y_col="value", series_col="series"
    )
    assert list(out) == ["m0", "m1"]
    assert out["m0"]["x"] == [0.0, 1.0]
    assert out["m0"]["mean"] == [2.0, 5.0]
    assert out["m0"]["std"] == [1.0, 0.0]
    assert out["m0"]["upper"] == [3.0, 5.0]
    assert out["m0"]["lower"] == [1.0, 5.0]
    assert out["m0"]["n"] == [2.0, 1.0]
    assert out["m1"]["mean"] == [7.0]


def test_table_to_line_series_arrays():
    assert mod._table_to_line_series_arrays(
        None, x_col="step", y_col="value", series_col="series"
    ) == ([], [], [])
    with pytest.raises(ValueError, match="missing required columns"):
        mod._table_to_line_series_arrays(
            _table([]), x_col="x", y_col="value", series_col="series"
        )
    table = _table(
        [
            [1, 0, 1, "b", 2.0],
            [0, 0, 0, "b", 1.0],
            [0, 0, 0, "a", 5.0],
            [None, 0, 0, "a", 1.0],
        ]
    )
    xs, ys, keys = mod._table_to_line_series_arrays(
        table, x_col="step", y_col="value", series_col="series"
    )
    assert keys == ["a", "b"]
    assert xs == [[0.0], [0.0, 1.0]]
    assert ys == [[5.0], [1.0, 2.0]]


def test_log_mechanism_shaded_error_plot(fake_run):
    mod._log_mechanism_shaded_error_plot(
        wandb_run=fake_run,
        prefix="p",
        plot_name="x",
        table=_table([]),
        x_col="step",
        y_col="value",
        series_col="series",
        step=0,
        max_rows=10,
        title="t",
    )
    assert fake_run.logs == []

    table = _table(
        [[0, 0, 0, M0_S1, 1.0], [0, 0, 0, M0_S2, 3.0], [0, 0, 0, M1_S1, 2.0]]
    )
    mod._log_mechanism_shaded_error_plot(
        wandb_run=fake_run,
        prefix="p",
        plot_name="x",
        table=table,
        x_col="step",
        y_col="value",
        series_col="series",
        step=5,
        max_rows=10,
        title="t",
    )
    payload, step, commit = fake_run.logs[0]
    assert step == 5 and commit is False
    fig = payload["p/plots/x"]
    assert isinstance(fig, go.Figure)
    assert [tr.name for tr in fig.data] == [
        "m0 ±1 std",
        "m0 mean",
        "m1 ±1 std",
        "m1 mean",
    ]
    assert fig.layout.title.text == "t"


def test_log_multiline_table_as_plot(fake_run):
    mod._log_multiline_table_as_plot(
        wandb_run=fake_run,
        prefix="p",
        plot_name="x",
        table=_table([]),
        x_col="step",
        y_col="value",
        series_col="series",
        step=0,
        max_rows=10,
        title="t",
    )
    assert fake_run.logs == [({}, 0, False)]
    mod._log_multiline_table_as_plot(
        wandb_run=fake_run,
        prefix="p",
        plot_name="x",
        table=_table([[0, 0, 0, "a", 1.0]]),
        x_col="step",
        y_col="value",
        series_col="series",
        step=1,
        max_rows=10,
        title="t",
    )
    assert "p/plots/x" in fake_run.logs[1][0]


def test_log_train_eval_return_by_mechanism_plot(fake_run):
    columns = ("step", "outer_iter", "train_step", "phase", "series", "value")
    mod._log_train_eval_return_by_mechanism_plot(
        wandb_run=fake_run,
        base_prefix="b",
        table=_table([], columns),
        x_col="step",
        y_col="value",
        series_col="series",
        phase_col="phase",
        step=0,
        max_rows=10,
    )
    assert fake_run.logs == []

    table = _table(
        [
            [0, 0, 0, "train", M0_S1, 1.0],
            [0, 0, 0, "train", M0_S2, 3.0],
            [0, 0, 0, "eval", M0_S1, 2.0],
            [0, 0, 0, "other", M1_S1, 2.0],
            [None, 0, 0, "eval", M0_S1, 2.0],
        ],
        columns,
    )
    mod._log_train_eval_return_by_mechanism_plot(
        wandb_run=fake_run,
        base_prefix="b",
        table=table,
        x_col="step",
        y_col="value",
        series_col="series",
        phase_col="phase",
        step=4,
        max_rows=10,
    )
    fig = fake_run.logs[0][0]["b/plots/returns/train_vs_eval_by_mechanism_mean_std"]
    names = [tr.name for tr in fig.data]
    assert names == [
        "eval m0 ±1 std",
        "eval m0 mean",
        "other m1 ±1 std",
        "other m1 mean",
        "train m0 ±1 std",
        "train m0 mean",
    ]
    assert fig.data[2].fillcolor == "rgba(0, 0, 0, 0.20)"
    assert list(fig.data[5].y) == [2.0]


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------


def test_extract_episode_metrics_newstack():
    out = mod.extract_episode_metrics_newstack(make_results())
    assert out["episode_return_mean"] == 12.0
    assert out["num_episodes_lifetime"] == 40.0
    legacy = mod.extract_episode_metrics_newstack(
        {"env_runners": {"episode_reward_mean": 3.0}}
    )
    assert legacy["episode_return_mean"] == 3.0
    assert legacy["episode_len_mean"] is None
    assert mod.extract_episode_metrics_newstack({})["episode_return_mean"] is None


def test_extract_series_returns_module_path():
    out = mod.extract_series_returns_newstack(make_results())
    # 200 module steps / 2 fisher agents = 100 env steps -> return / 100
    assert out == {M0_S1: 0.1, M0_S2: 0.2, M1_S1: 0.3}
    # "odd_module" has no "_policy" suffix and is dropped


def test_extract_series_returns_module_path_edge_cases():
    results = make_results()
    env = results["env_runners"]
    env["num_module_steps_sampled"][M0_S1] = 0  # module_env_steps <= 0
    env["module_episode_returns_mean"][M0_S2] = float("nan")
    out = mod.extract_series_returns_newstack(results)
    assert out == {M1_S1: 0.3}

    # no agent steps at all -> agent count unknown -> nothing normalised
    env["num_agent_steps_sampled"] = {}
    assert mod.extract_series_returns_newstack(results) == {}

    # agents of another type only
    env["num_agent_steps_sampled"] = {"merchant:0": 10}
    assert mod.extract_series_returns_newstack(results) == {}


def test_extract_series_returns_agent_fallback_and_empty():
    results = {
        "env_runners": {
            "agent_episode_returns_mean": {
                "a": 10.0,
                "b": 5.0,
                "c": float("nan"),
                "d": 1.0,
            },
            "num_agent_steps_sampled": {"a": 100, "b": 0, "c": 10, "d": None},
        }
    }
    assert mod.extract_series_returns_newstack(results) == {"a": 0.1}
    assert mod.extract_series_returns_newstack({}) == {}
    assert mod.extract_series_returns_newstack({"env_runners": None}) == {}


def test_extract_perf_newstack():
    out = mod.extract_perf_newstack(make_results())
    assert out["env_steps_this_iter"] == 100.0
    assert out["env_steps_lifetime"] == 1000.0
    assert out["agent_steps_this_iter_sum"] == 200.0
    assert out["agent_steps_lifetime_sum"] == 2000.0
    assert out["env_steps_throughput"] == 50.0
    assert out["training_iteration_s"] == 1.5
    assert out["sample_s"] == 0.5
    assert out["training_step_s"] is None
    assert out["weights_seq_no"] == 3.0

    restore = mod.extract_perf_newstack(
        {
            "env_runners": {
                "num_env_steps_sampled_lifetime_throughput": {
                    "throughput_since_last_restore": 7.0
                }
            }
        }
    )
    assert restore["env_steps_throughput"] == 7.0
    empty = mod.extract_perf_newstack({})
    assert all(v is None for v in empty.values())


def test_extract_learner_metrics_newstack():
    out = mod.extract_learner_metrics_newstack(make_results())
    assert set(out) == {M0_S1, M0_S2, M1_S1, "__all_modules__"}
    m = out[M0_S1]
    assert m["total_loss"] == 1.0
    assert "nested" not in m and "listed" not in m and "nan_metric" not in m
    assert m["module_steps_throughput_since_last_reduce"] == 10.0
    assert m["module_steps_throughput_since_last_restore"] == 11.0
    assert m["policy_relative_entropy"] == pytest.approx(80.0)
    assert m["entropy_pressure"] == pytest.approx(0.008)
    # lag1 (3) + training calls since sync (2) + outstanding reqs (1) + queue wait (0.5)
    assert m["sample_staleness"] == pytest.approx(6.5)

    all_modules = out["__all_modules__"]
    assert all_modules["policy_relative_entropy"] is None
    assert "entropy_pressure" not in all_modules
    assert all_modules["sample_staleness"] == pytest.approx(3.5)

    assert mod.extract_learner_metrics_newstack({}) == {}
    assert mod.extract_learner_metrics_newstack({"learners": "bad"}) == {}
    minimal = mod.extract_learner_metrics_newstack({"learners": {"p": {}}})
    assert minimal["p"]["sample_staleness"] is None


# ---------------------------------------------------------------------------
# plot_training_results_new_stack
# ---------------------------------------------------------------------------


def test_plot_training_results_early_returns(fake_run):
    mod.plot_training_results_new_stack(
        None, outer_iter=0, training_episode=0, results={}
    )
    mod.plot_training_results_new_stack(
        fake_run, outer_iter=0, training_episode=0, results=None
    )
    assert fake_run.logs == []


def test_plot_training_results_defaults(fake_run):
    mod.plot_training_results_new_stack(
        fake_run,
        outer_iter=1,
        training_episode=2,
        results=make_results(),
        prefix="rllib",
    )
    gs = 1_000_002
    metrics = fake_run.logs[0][0]
    assert metrics["rllib/outer_iter"] == 1
    assert metrics["rllib/train_step"] == 2
    assert metrics["rllib/perf/env_steps_this_iter"] == 100.0
    assert metrics["rllib/water/reservoir_stage_m_mean"] == 4.0
    assert metrics["rllib/water/total_usage_m3s_max"] == 2.0
    assert "rllib/perf/training_step_s" not in metrics  # None dropped
    assert "rllib/rllib_raw/episode_return_mean" not in metrics
    assert metrics["rllib/series/reward_per_agent_per_step_mean"] == pytest.approx(0.2)
    assert not any(
        k.startswith("rllib/series/reward_per_agent_per_step/") for k in metrics
    )
    assert not any(k.startswith("rllib/learner/") for k in metrics)

    # tables cached but no plots logged with the defaults
    assert len(mod._RETURNS_TABLES[(id(fake_run), "rllib")].data) == 3
    per_metric = mod._LEARNER_METRIC_TABLES[(id(fake_run), "rllib")]
    assert set(per_metric) == {
        "total_loss",
        "policy_loss",
        "vf_loss",
        "entropy",
        "sample_staleness",
    }
    assert len(per_metric["total_loss"].data) == 3  # __all_modules__ excluded
    assert not any(k.startswith("rllib/plots/") for k in fake_run.logged_keys())
    assert fake_run.logs[-1] == ({}, gs, False)
    assert all(step == gs for _, step, _ in fake_run.logs)


def test_plot_training_results_all_flags(fake_run):
    mod.plot_training_results_new_stack(
        fake_run,
        outer_iter=0,
        training_episode=1,
        results=make_results(),
        prefix="rllib",
        include_all_modules_in_learner_plots=True,
        skip_learner_plot_keys={"vf_loss"},
        learner_plot_whitelist={"total_loss", "vf_loss"},
        log_per_policy_learner_scalars=True,
        learner_scalar_whitelist={"entropy"},
        log_per_series_return_scalars=True,
        log_return_multiline_plot=True,
        log_learner_multiline_plots=True,
        log_mechanism_shaded_plots=True,
        log_raw_rllib_episode_metrics=True,
    )
    metrics = fake_run.logs[0][0]
    assert metrics["rllib/rllib_raw/episode_return_mean"] == 12.0
    assert metrics[f"rllib/series/reward_per_agent_per_step/{M0_S1}"] == pytest.approx(
        0.1
    )
    assert metrics[f"rllib/learner/{M0_S1}/entropy"] == 0.8
    assert f"rllib/learner/{M0_S1}/total_loss" not in metrics  # scalar whitelist wins

    keys = fake_run.logged_keys()
    assert "rllib/plots/returns/all_series_return_mean" in keys
    assert "rllib/plots/returns/by_mechanism_mean_std" in keys
    assert "rllib/plots/learner/total_loss" in keys
    assert "rllib/plots/learner/total_loss_by_mechanism_mean_std" in keys
    assert not any("vf_loss" in k for k in keys)  # skipped even though whitelisted

    per_metric = mod._LEARNER_METRIC_TABLES[(id(fake_run), "rllib")]
    assert set(per_metric) == {"total_loss"}
    assert len(per_metric["total_loss"].data) == 4  # __all_modules__ included


def test_plot_training_results_train_eval_prefixes(fake_run):
    results = make_results(with_learners=False)
    mod.plot_training_results_new_stack(
        fake_run,
        outer_iter=0,
        training_episode=0,
        results=results,
        prefix="rllib/train",
        log_mechanism_shaded_plots=True,
    )
    combo_key = (id(fake_run), "rllib")
    assert len(mod._TRAIN_EVAL_RETURN_TABLES[combo_key].data) == 3
    assert (
        "rllib/plots/returns/train_vs_eval_by_mechanism_mean_std"
        not in fake_run.logged_keys()
    )
    # shaded per-prefix plot is suppressed for phase-suffixed prefixes
    assert (
        "rllib/train/plots/returns/by_mechanism_mean_std" not in fake_run.logged_keys()
    )

    mod.plot_training_results_new_stack(
        fake_run, outer_iter=0, training_episode=0, results=results, prefix="rllib/eval"
    )
    assert len(mod._TRAIN_EVAL_RETURN_TABLES[combo_key].data) == 6
    assert (
        "rllib/plots/returns/train_vs_eval_by_mechanism_mean_std"
        in fake_run.logged_keys()
    )
    assert (id(fake_run), "rllib/train") in mod._RETURNS_TABLES
    assert (id(fake_run), "rllib/eval") in mod._RETURNS_TABLES
    assert (id(fake_run), "rllib/eval") not in mod._LEARNER_METRIC_TABLES


def test_plot_training_results_without_series_or_learners(fake_run):
    results = {"env_runners": {"episode_return_mean": 1.0}}
    mod.plot_training_results_new_stack(
        fake_run, outer_iter=0, training_episode=0, results=results
    )
    assert mod._RETURNS_TABLES == {}
    assert mod._LEARNER_METRIC_TABLES == {}
    assert len(fake_run.logs) == 2
