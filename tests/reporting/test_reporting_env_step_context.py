"""Unit tests for ``core.reporting.utils.env_step_context``.

The module turns one ``EnvStepContext`` into per-agent series (reward,
action, observation, info), accumulates them in per-run ``wandb.Table``
caches and logs one ``line_series`` plot per series. The tests exercise the
extractors on every payload shape they accept (``None``, scalar, dict of
scalars, dict of vectors, with and without an ``observation_map``) and the
logging entry point with a fake run.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import wandb
from core.reporting.utils import env_step_context as mod
from core.world.context import Context, ContextSchema, MechanismStatus
from tests.reporting.conftest import make_env_ctx

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------


def test_extract_reward_series_shapes():
    assert mod._extract_reward_series(None) == {}
    assert mod._extract_reward_series({"a": 1, 2: np.float32(2.5)}) == {
        "a": 1.0,
        "2": 2.5,
    }
    assert mod._extract_reward_series(3) == {"global": 3.0}


def test_extract_action_series_shapes():
    assert mod._extract_action_series(None) == {}
    out = mod._extract_action_series({"a": 0.5, "b": np.array([1.0, 2.0])})
    assert out == {"a": 0.5, "b/dim_0": 1.0, "b/dim_1": 2.0}
    assert mod._extract_action_series(np.float64(7.0)) == {"global": 7.0}
    assert mod._extract_action_series([1.0, 2.0]) == {
        "global/dim_0": 1.0,
        "global/dim_1": 2.0,
    }


def test_extract_observation_series_non_dict():
    assert mod._extract_observation_series(None) == {}
    assert mod._extract_observation_series(1.5) == {"observation": {"global": 1.5}}
    assert mod._extract_observation_series(1.5, observation_map=["fish"]) == {
        "fish": {"global": 1.5}
    }
    # vector with a partial map: the mapped names first, generic names after
    out = mod._extract_observation_series([1.0, 2.0, 3.0], observation_map=["a", "b"])
    assert out == {"a": {"global": 1.0}, "b": {"global": 2.0}, "obs_2": {"global": 3.0}}


def test_extract_observation_series_dict_of_dicts():
    obs = {"ag": {"fish": 1.0, "vec": np.array([2.0, 3.0])}}
    out = mod._extract_observation_series(obs, observation_map=["x"])
    assert out == {"fish": {"ag": 1.0}, "x": {"ag": 2.0}, "vec_1": {"ag": 3.0}}


def test_extract_observation_series_dict_of_values():
    out = mod._extract_observation_series({"a": 1.0, "b": np.array([2.0])})
    assert out == {"observation": {"a": 1.0, "b": 2.0}}
    out = mod._extract_observation_series({"a": 1.0}, observation_map=["fish"])
    assert out == {"fish": {"a": 1.0}}
    out = mod._extract_observation_series(
        {"a": np.array([1.0, 2.0])}, observation_map=["fish"]
    )
    assert out == {"fish": {"a": 1.0}, "obs_1": {"a": 2.0}}


def test_extract_info_series_filters_non_numeric():
    assert mod._extract_info_series(None) == {}
    assert mod._extract_info_series([1, 2]) == {}
    info = {
        "ag": {"fish": 1.0, "label": "abc", "nested": {"k": 1}, "vec": [1.0, 2.0]},
        "skipped": 3.0,
    }
    out = mod._extract_info_series(info)
    assert out == {"fish": {"ag": 1.0}, "vec_0": {"ag": 1.0}, "vec_1": {"ag": 2.0}}


# ---------------------------------------------------------------------------
# table -> line_series conversion
# ---------------------------------------------------------------------------


def test_table_to_line_series_arrays_groups_and_sorts():
    assert mod._table_to_line_series_arrays(
        None, x_col="x", y_col="y", series_col="s"
    ) == ([], [], [])

    table = SimpleNamespace(
        columns=["env_step", "agent_id", "value"],
        data=[[2, "b", 1.0], [1, "b", 0.5], [0, "a", 3.0], [None, "a", 1.0]],
    )
    xs, ys, keys = mod._table_to_line_series_arrays(
        table, x_col="env_step", y_col="value", series_col="agent_id"
    )
    assert keys == ["a", "b"]
    assert xs == [[0.0], [1.0, 2.0]]
    assert ys == [[3.0], [0.5, 1.0]]


def test_log_multiline_table_as_plot_empty_table_logs_empty_payload(fake_run):
    table = wandb.Table(columns=["env_step", "agent_id", "value"])
    mod._log_multiline_table_as_plot(
        wandb_run=fake_run,
        plot_key="p/plot",
        table=table,
        x_col="env_step",
        y_col="value",
        series_col="agent_id",
        step=1,
        title="t",
    )
    assert fake_run.logs == [({}, 1, False)]


# ---------------------------------------------------------------------------
# plot_env_step_context
# ---------------------------------------------------------------------------


def test_plot_env_step_context_early_returns(fake_run, env_ctx):
    mod.plot_env_step_context(None, ctx=env_ctx)
    mod.plot_env_step_context(fake_run, ctx=None)
    bad = Context(id="x", opt_id="o", step=0, env="e", payload=ContextSchema())
    mod.plot_env_step_context(fake_run, ctx=bad)
    assert fake_run.logs == []


def test_plot_env_step_context_logs_every_series(fake_run):
    ctx = make_env_ctx(step=4, observation_map=["fish", "quota"])
    mod.plot_env_step_context(
        fake_run,
        ctx=ctx,
        prefix="env",
        obs_keys_skip={"quota"},
        infos_keys_skip={"fish"},
    )

    keys = fake_run.logged_keys()
    assert {"env/mechanism/index", "env/seed", "env/status"} <= keys
    assert fake_run.payload_for("env/status") == MechanismStatus.train.value
    assert "env/plots/reward" in keys
    assert "env/plots/action" in keys
    assert "env/plots/observation/fish" in keys
    assert "env/plots/observation/quota" not in keys
    assert "env/plots/info/fish" not in keys
    assert {"env/plots/info/H_realized", "env/plots/info/farm_area_m2"} <= keys

    # the final commit closes the step
    assert fake_run.logs[-1] == ({}, 4, True)

    run_key = id(fake_run)
    assert len(mod._ENV_REWARD_TABLES[run_key].data) == 2
    assert len(mod._ENV_ACTION_TABLES[run_key].data) == 2
    assert set(mod._ENV_OBS_TABLES[run_key]) == {"fish"}  # skipped keys get no table
    assert set(mod._ENV_INFO_TABLES[run_key]) == {"H_realized", "farm_area_m2"}


def test_plot_env_step_context_accumulates_across_steps(fake_run):
    for step in range(3):
        mod.plot_env_step_context(fake_run, ctx=make_env_ctx(step=step))
    assert len(mod._ENV_REWARD_TABLES[id(fake_run)].data) == 6
    plot = fake_run.logs[-2][0]  # last info plot before the commit
    assert len(plot) == 1


def test_plot_env_step_context_without_mechanism_or_seed(fake_run):
    ctx = make_env_ctx(step=0, mechanism=None, seed=None, reward=0.0)
    ctx.payload.action = None  # the reset record carries no action
    mod.plot_env_step_context(fake_run, ctx=ctx)
    keys = fake_run.logged_keys()
    assert "env/mechanism/index" not in keys
    assert "env/seed" not in keys
    assert "env/status" in keys
    assert "env/plots/reward" in keys  # scalar reward -> "global" series
    assert "env/plots/action" not in keys
