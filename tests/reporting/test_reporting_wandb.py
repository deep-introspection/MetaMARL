"""Unit tests for ``core.reporting.wandb.WandbReporter``.

``WandbReporter`` is a Ray actor that owns the single ``wandb.Run`` and
delegates every plot to the utilities in ``core.reporting.utils``. The tests
instantiate the undecorated class (``__ray_metadata__.modified_class``) with
``wandb.init`` replaced by a fake run, then check metric definitions, direct
logging, the ``finish`` idempotence and the argument forwarding of each
``plot_*`` entry point to a spied utility.
"""

from __future__ import annotations

import numpy as np
import pytest

import wandb
from core.reporting import wandb as reporter_mod
from core.reporting.utils.env_reduced import ReductionSpec
from tests.reporting.conftest import make_env_ctx

pytestmark = pytest.mark.unit

ReporterClass = reporter_mod.WandbReporter.__ray_metadata__.modified_class


@pytest.fixture
def reporter(monkeypatch, fake_run):
    init_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return fake_run

    monkeypatch.setattr(wandb, "init", fake_init)
    rep = ReporterClass(project="proj", name="run", config={"a": 1})
    rep._init_calls = init_calls
    return rep


def test_init_opens_one_run(reporter, fake_run):
    assert reporter._run is fake_run
    (kwargs,) = reporter._init_calls
    assert kwargs["project"] == "proj"
    assert kwargs["name"] == "run"
    assert kwargs["config"] == {"a": 1}
    assert kwargs["reinit"] is True
    assert isinstance(kwargs["settings"], wandb.Settings)


def test_init_defaults_config_and_settings(monkeypatch, fake_run):
    captured = {}
    monkeypatch.setattr(wandb, "init", lambda **kw: captured.update(kw) or fake_run)
    ReporterClass(project="p", name="n", settings={"mode": "offline"})
    assert captured["config"] == {}
    assert captured["settings"].mode == "offline"


def test_ensure_prefix_metrics_defines_once(reporter, fake_run):
    reporter._ensure_prefix_metrics("env")
    reporter._ensure_prefix_metrics("env")
    assert fake_run.defined == [
        ("env/train_step", {}),
        ("env/*", {"step_metric": "env/train_step"}),
    ]


def test_define_metric_forwards_only_given_kwargs(reporter, fake_run):
    reporter.define_metric("x")
    reporter.define_metric("y", step_metric="s", hidden=True, summary="max")
    assert fake_run.defined == [
        ("x", {}),
        ("y", {"step_metric": "s", "hidden": True, "summary": "max"}),
    ]


def test_log_and_log_many(reporter, fake_run):
    reporter.log({"a": 1}, step=3)
    reporter.log_many([{"payload": {"b": 2}, "step": 4}, {"payload": {"c": 3}}])
    assert fake_run.logs == [
        ({"a": 1}, 3, None),
        ({"b": 2}, 4, None),
        ({"c": 3}, None, None),
    ]


def test_finish_is_idempotent(reporter, fake_run):
    reporter.finish()
    reporter.finish()
    assert fake_run.finished == 1
    assert reporter._run is None


def test_plot_ray_result_forwards_and_masks_learner_plots_on_eval(
    monkeypatch, reporter, fake_run
):
    calls = []
    monkeypatch.setattr(
        reporter_mod, "plot_training_results_new_stack", lambda **kw: calls.append(kw)
    )
    results = {"env_runners": {}}

    reporter.plot_ray_result(
        1, 2, results, prefix="rllib/train", log_learner_multiline_plots=True
    )
    reporter.plot_ray_result(
        1, 3, results, prefix="rllib/eval", log_learner_multiline_plots=True
    )

    assert fake_run.defined[0] == ("rllib/train/train_step", {})
    train, eval_ = calls
    assert train["wandb_run"] is fake_run
    assert train["outer_iter"] == 1 and train["training_episode"] == 2
    assert train["results"] is results
    assert train["log_learner_multiline_plots"] is True
    assert eval_["log_learner_multiline_plots"] is False
    # the reporter always asks for the raw RLlib episode metrics
    assert train["log_raw_rllib_episode_metrics"] is True
    assert train["max_lines_returns"] == 64
    assert train["log_mechanism_shaded_plots"] is True


def test_plot_env_step_forwards(monkeypatch, reporter, fake_run):
    calls = []
    monkeypatch.setattr(
        reporter_mod, "plot_env_step_context", lambda **kw: calls.append(kw)
    )
    ctx = make_env_ctx(step=2)
    reporter.plot_env_step(ctx=ctx, prefix="env", obs_keys_skip={"q"})
    assert calls == [
        {"wandb_run": fake_run, "ctx": ctx, "prefix": "env", "obs_keys_skip": {"q"}}
    ]
    assert ("env/train_step", {}) in fake_run.defined


def test_plot_env_reduced_forwards(monkeypatch, reporter, fake_run):
    calls = []
    monkeypatch.setattr(reporter_mod, "plot_env_reduced", lambda **kw: calls.append(kw))
    ctxs = [make_env_ctx(step=0)]
    reducers = [ReductionSpec(name="auto_env_metrics")]
    reporter.plot_env_reduced(
        ctxs=ctxs, outer_iter=1, training_episode=2, reducers=reducers
    )
    (kw,) = calls
    assert kw == {
        "wandb_run": fake_run,
        "ctxs": ctxs,
        "outer_iter": 1,
        "training_episode": 2,
        "reducers": reducers,
        "prefix": "env_reduced",
    }
    assert ("env_reduced/train_step", {}) in fake_run.defined


def test_plot_es_population_defines_generation_metric_once(
    monkeypatch, reporter, fake_run
):
    calls = []
    monkeypatch.setattr(
        reporter_mod, "plot_es_population_util", lambda **kw: calls.append(kw)
    )
    population = np.ones((1, 2))
    fitness = np.array([0.5])
    for generation in range(2):
        reporter.plot_es_population(
            generation=generation,
            population=population,
            fitness=fitness,
            parameter_names=["a", "b"],
            sigma=0.1,
        )
    assert fake_run.defined == [
        ("es/generation", {}),
        ("es/*", {"step_metric": "es/generation"}),
    ]
    assert len(calls) == 2
    assert calls[1]["generation"] == 1
    assert calls[1]["wandb_run"] is fake_run
    assert calls[1]["sigma"] == 0.1
    assert calls[1]["mean"] is None
    assert calls[1]["prefix"] == "es"


def test_plot_es_population_reaches_real_utility(reporter, fake_run):
    reporter.plot_es_population(
        generation=0,
        population=np.array([[1.0, 2.0]]),
        fitness=np.array([-0.1]),
        parameter_names=["fixed_quota", "fine_amount"],
    )
    assert "es/plots/parallel_coordinates" in fake_run.logged_keys()
