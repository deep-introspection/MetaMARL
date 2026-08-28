"""Unit tests for ``BilevelConfig.build_optimizer`` and the reporter hand-off.

The composition root starts Ray, creates the reporter and ``World`` actors and
builds both levels. Here ``RayRuntime.ensure_initialized``, ``WandbReporter``
and ``World`` are patched in the ``core.optimizers.bilevel`` namespace, and the
inner/outer configs are replaced by recording stubs, so the wiring (mechanism
template, seed propagation, population sizing) is checked without any Ray
runtime. The ``None`` branches of the fluent setters and the reporter
``finish`` call at the end of ``BilevelOptimizer.run`` are covered as well.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import ray

import core.optimizers.bilevel as bilevel_module
from core.adaptors.ray.runtime import RayRuntimeConfig
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer


class StubOptimizer:
    """Built optimizer exposing only what the composition root touches."""

    def __init__(self, batch_capacity=None):
        self._batch_capacity = batch_capacity
        self.best_fitness = 1.0
        self.best_candidate = np.array([0.5])

    @property
    def batch_capacity(self):
        return self._batch_capacity

    @batch_capacity.setter
    def batch_capacity(self, value):
        self._batch_capacity = value

    def run(self):
        return {"best_fitness": self.best_fitness}


class StubLevelConfig:
    """Recording stand-in for an inner or outer ``OptimizerConfig``.

    ``copy`` returns ``self`` so the test can inspect what the composition root
    wrote into the copy; ``build_optimizer`` records its keyword arguments.
    """

    def __init__(self, *, seeds=None, eval_seeds=None, batch_capacity=None):
        self.env_config: dict = {}
        self.seeds = seeds
        self.eval_seeds = eval_seeds
        self.dimension = None
        self.build_kwargs: dict | None = None
        self.built = StubOptimizer(batch_capacity)

    def copy(self, copy_frozen=None):
        return self

    def _merge_env_config(self, extra):
        self.env_config = {**self.env_config, **extra}
        return self

    def build_optimizer(self, **kwargs):
        self.build_kwargs = kwargs
        return self.built


class StubActorClass:
    """Mimics ``Actor.options(name=...).remote(**kwargs)`` of a Ray actor class."""

    def __init__(self, handle):
        self.handle = handle
        self.options_kwargs: dict | None = None
        self.remote_kwargs: dict | None = None

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, **kwargs):
        self.remote_kwargs = kwargs
        return self.handle


@pytest.fixture
def patched_actors(monkeypatch):
    """Patch Ray init, ``World`` and ``WandbReporter`` in the bilevel module."""
    init_calls = []
    monkeypatch.setattr(
        bilevel_module.RayRuntime,
        "ensure_initialized",
        classmethod(lambda cls, cfg: init_calls.append(cfg)),
    )
    world_handle = SimpleNamespace(kind="world")
    reporter_handle = SimpleNamespace(kind="reporter")
    world_cls = StubActorClass(world_handle)
    reporter_cls = StubActorClass(reporter_handle)
    monkeypatch.setattr(bilevel_module, "World", world_cls)
    monkeypatch.setattr(bilevel_module, "WandbReporter", reporter_cls)
    return SimpleNamespace(
        init_calls=init_calls,
        world_cls=world_cls,
        reporter_cls=reporter_cls,
        world_handle=world_handle,
        reporter_handle=reporter_handle,
    )


def make_config(inner, outer, mechanism=None):
    return (
        BilevelConfig()
        .world(world_name="fishery")
        .mechanism(mechanism=mechanism or SubsidyMechanism(subsidy=0.1, cost=0.1))
        .training(outer_iters=5)
        .inner(inner)
        .outer(outer)
    )


@pytest.mark.unit
def test_build_wires_world_mechanism_seeds_and_population(patched_actors):
    inner = StubLevelConfig(seeds=[1, 2], eval_seeds=[9], batch_capacity=6)
    outer = StubLevelConfig()
    mechanism = SubsidyMechanism(subsidy=0.1, cost=0.1)
    cfg = make_config(inner, outer, mechanism)

    opt = cfg.build_optimizer()

    # Ray started with a default runtime config; no reporter without wandb.
    assert len(patched_actors.init_calls) == 1
    assert isinstance(patched_actors.init_calls[0], RayRuntimeConfig)
    assert cfg.reporter is None
    assert patched_actors.reporter_cls.remote_kwargs is None

    # World actor named after the config and given the (absent) reporter.
    assert patched_actors.world_cls.options_kwargs == {"name": cfg.world_name}
    assert patched_actors.world_cls.remote_kwargs == {"reporting": None}

    # Mechanism template and seeds flow into both levels.
    assert outer.dimension == mechanism.dimension
    assert inner.env_config == {"mechanism": mechanism}
    assert outer.env_config == {
        "seeds": [1, 2],
        "eval_seeds": [9],
        "mechanism": mechanism,
    }
    assert inner.build_kwargs == {
        "world": patched_actors.world_handle,
        "world_name": cfg.world_name,
        "reporting": None,
    }
    assert outer.build_kwargs == {
        "world": patched_actors.world_handle,
        "inner_opt": inner.built,
        "reporting": None,
    }

    # Population size copied from the inner batch capacity.
    assert outer.built.batch_capacity == 6
    assert isinstance(opt, BilevelOptimizer)
    assert opt.outer is outer.built and opt.inner is inner.built
    assert opt.max_outer_iters == 5
    assert opt.world_name == cfg.world_name


@pytest.mark.unit
def test_build_skips_seed_propagation_when_inner_has_none(patched_actors):
    inner = StubLevelConfig(seeds=None, eval_seeds=None, batch_capacity=2)
    outer = StubLevelConfig()
    mechanism = SubsidyMechanism(subsidy=0.1, cost=0.1)

    make_config(inner, outer, mechanism).build_optimizer()

    assert outer.env_config == {"mechanism": mechanism}


@pytest.mark.unit
def test_build_creates_wandb_reporter_and_passes_it_down(patched_actors):
    inner = StubLevelConfig(batch_capacity=3)
    outer = StubLevelConfig()
    ray_cfg_marker = RayRuntimeConfig(num_cpus=1)
    cfg = make_config(inner, outer).reporting(
        "wandb",
        project_name="proj",
        config={"tag": "x"},
        settings_dict={"mode": "offline"},
    )
    cfg.ray_cfg = ray_cfg_marker

    cfg.build_optimizer()

    assert patched_actors.init_calls == [ray_cfg_marker]
    assert cfg.reporter is patched_actors.reporter_handle
    assert patched_actors.reporter_cls.options_kwargs == {
        "name": f"{cfg.world_name}_wandb"
    }
    assert patched_actors.reporter_cls.remote_kwargs == {
        "project": "proj",
        "name": f"proj-{cfg.world_name}",
        "config": {"outer_iters": 5, "world_name": cfg.world_name, "tag": "x"},
        "settings": {"mode": "offline"},
    }
    assert patched_actors.world_cls.remote_kwargs == {
        "reporting": patched_actors.reporter_handle
    }
    assert inner.build_kwargs["reporting"] is patched_actors.reporter_handle
    assert outer.build_kwargs["reporting"] is patched_actors.reporter_handle


@pytest.mark.unit
def test_build_requires_mechanism_and_both_levels(patched_actors):
    with pytest.raises(ValueError, match="mechanism"):
        BilevelConfig().inner(StubLevelConfig()).outer(
            StubLevelConfig()
        ).build_optimizer()
    with pytest.raises(ValueError, match="inner"):
        BilevelConfig().mechanism(
            mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1)
        ).inner(StubLevelConfig()).build_optimizer()
    assert patched_actors.init_calls == []


@pytest.mark.unit
def test_setters_ignore_none_and_keep_previous_values():
    mechanism = SubsidyMechanism(subsidy=0.1, cost=0.1)
    inner, outer = StubLevelConfig(), StubLevelConfig()
    cfg = make_config(inner, outer, mechanism)
    world_name = cfg.world_name

    cfg.inner(None).outer(None).world(world_name=None).mechanism(mechanism=None)
    cfg.training(outer_iters=None, output_dir="out")

    assert cfg.inner_cfg is inner and cfg.outer_cfg is outer
    assert cfg.world_name == world_name
    assert cfg.mechanism_template is mechanism
    assert cfg.outer_iters == 5
    assert cfg.output_dir == "out"


@pytest.mark.unit
def test_world_name_gets_a_random_suffix():
    a = BilevelConfig().world(world_name="fishery").world_name
    b = BilevelConfig().world(world_name="fishery").world_name
    assert a.startswith("fishery_") and b.startswith("fishery_")
    assert a != b


@pytest.mark.unit
def test_run_finishes_reporter_when_present(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    finish_calls = []
    cfg = make_config(StubLevelConfig(), StubLevelConfig()).training(outer_iters=2)
    cfg._reporter = SimpleNamespace(
        finish=SimpleNamespace(remote=lambda: finish_calls.append("finish"))
    )
    outer = StubOptimizer()

    result = BilevelOptimizer(cfg, outer=outer, inner=object()).run()

    assert finish_calls == ["finish"]
    assert result["outer_iters"] == 2
    assert result["best_fitness"] == 1.0
