"""Unit tests for ``BilevelConfig.build_optimizer`` (no Ray runtime).

The composition root starts Ray, creates the ``World`` actor, builds the
primary reporter from ``reporter_cfg`` and wires both levels. Here
``RayRuntime.ensure_initialized`` and ``World`` are patched in the
``core.optimizers.bilevel`` namespace, the reporter config is a recording
stub and the inner/outer configs are recording stand-ins, so the wiring
(reporter hand-off, mechanism space, seed propagation, population sizing) is
checked without any actor.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import core.optimizers.bilevel as bilevel_module
from core.adaptors.ray.runtime import RayRuntimeConfig
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer
from core.reporting.config import ReporterConfig


class StubOptimizer:
    """Built optimizer exposing only what the composition root touches."""

    def __init__(self, batch_capacity=None):
        self.batch_capacity = batch_capacity
        self.best_fitness = 1.0
        self.best_candidate = np.array([0.5])

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
        self.reporter_cfg = None
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


class RecordingReporterConfig(ReporterConfig):
    """Reporter config whose ``build`` returns a label-recording reporter."""

    def build(self, *, label=None):
        return SimpleNamespace(label=label, world=self.world, closed=False)


class StubSpace:
    dimension = 3

    @classmethod
    def default(cls):
        return "default-mechanism"


@pytest.fixture
def patched_actors(monkeypatch):
    """Patch Ray init and ``World`` in the bilevel module."""
    init_calls = []
    monkeypatch.setattr(
        bilevel_module.RayRuntime,
        "ensure_initialized",
        classmethod(lambda cls, cfg: init_calls.append(cfg)),
    )
    world_handle = SimpleNamespace(kind="world")
    world_cls = StubActorClass(world_handle)
    monkeypatch.setattr(bilevel_module, "World", world_cls)
    return SimpleNamespace(
        init_calls=init_calls, world_cls=world_cls, world_handle=world_handle
    )


def make_config(inner, outer, space=StubSpace):
    cfg = (
        BilevelConfig()
        .world(world_name="fishery")
        .training(outer_iters=5)
        .inner(inner)
        .outer(outer)
        .reporter(RecordingReporterConfig(project="p"))
    )
    if space is not None:
        cfg.mechanism(space=space)
    return cfg


@pytest.mark.unit
def test_build_wires_world_reporter_space_seeds_and_population(patched_actors):
    inner = StubLevelConfig(seeds=[1, 2], eval_seeds=[9], batch_capacity=6)
    outer = StubLevelConfig()
    cfg = make_config(inner, outer)

    opt = cfg.build_optimizer()

    # Ray started with a default runtime config, World named after the config.
    assert len(patched_actors.init_calls) == 1
    assert isinstance(patched_actors.init_calls[0], RayRuntimeConfig)
    assert patched_actors.world_cls.options_kwargs == {"name": cfg.world_name}
    assert patched_actors.world_cls.remote_kwargs == {}

    # Reporter config stamped with the run identity, then copied to each level.
    assert cfg.reporter_cfg.world == cfg.world_name
    assert cfg.reporter_cfg.outer_iters == 5
    assert opt.reporting.label == "bilvel"  # sic: label typo in production code
    assert opt.reporting.world == cfg.world_name
    for level in (inner, outer):
        assert isinstance(level.reporter_cfg, RecordingReporterConfig)
        assert level.reporter_cfg is not cfg.reporter_cfg
        assert level.reporter_cfg.world == cfg.world_name
    assert inner.reporter_cfg is not outer.reporter_cfg

    # Mechanism space and seeds flow into both levels.
    assert outer.dimension == 3
    assert inner.env_config == {"mechanism_space": StubSpace}
    assert outer.env_config == {
        "seeds": [1, 2],
        "eval_seeds": [9],
        "mechanism_space": StubSpace,
        "default_mechanism": "default-mechanism",
    }
    assert inner.build_kwargs == {
        "world": patched_actors.world_handle,
        "world_name": cfg.world_name,
    }
    assert outer.build_kwargs == {
        "world": patched_actors.world_handle,
        "inner_opt": inner.built,
    }

    # Population size copied from the inner batch capacity.
    assert outer.built.batch_capacity == 6
    assert isinstance(opt, BilevelOptimizer)
    assert opt.outer is outer.built and opt.inner is inner.built
    assert opt.max_outer_iters == 5
    assert opt.world_name == cfg.world_name
    assert opt.mechanism_space is StubSpace
    assert opt.config is cfg


@pytest.mark.unit
def test_build_without_space_or_seeds_only_forwards_none_values(patched_actors):
    inner = StubLevelConfig(seeds=None, eval_seeds=None, batch_capacity=2)
    outer = StubLevelConfig()

    make_config(inner, outer, space=None).build_optimizer()

    assert outer.dimension is None
    assert inner.env_config == {}
    assert outer.env_config == {"mechanism_space": None, "default_mechanism": None}
    assert outer.built.batch_capacity == 2


@pytest.mark.unit
def test_build_uses_configured_ray_runtime(patched_actors):
    cfg = make_config(StubLevelConfig(), StubLevelConfig()).ray(
        device="cpu", num_cpus=2, omp_threads=4, include_dashboard=False
    )

    cfg.build_optimizer()

    assert patched_actors.init_calls == [cfg.ray_cfg]
    assert cfg.ray_cfg.num_cpus == 2 and cfg.ray_cfg.omp_threads == 4
    assert cfg.ray_cfg.init_kwargs == {"include_dashboard": False}


@pytest.mark.unit
def test_build_requires_reporter_and_both_levels(patched_actors):
    """Document current behaviour: missing pieces fail with ``AttributeError``.

    ``build_optimizer`` dereferences ``inner_cfg``, ``outer_cfg`` and
    ``reporter_cfg`` without checking them, so a config built without one of
    them raises ``AttributeError`` on ``None`` instead of a ``ValueError``
    naming the missing piece.
    """
    with pytest.raises(AttributeError, match="copy"):
        BilevelConfig().outer(StubLevelConfig()).build_optimizer()
    with pytest.raises(AttributeError, match="copy"):
        BilevelConfig().inner(StubLevelConfig()).build_optimizer()
    with pytest.raises(AttributeError, match="world"):
        BilevelConfig().inner(StubLevelConfig()).outer(
            StubLevelConfig()
        ).build_optimizer()
    # Ray is started before the checks would have run
    assert len(patched_actors.init_calls) == 3
