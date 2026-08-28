"""Builder-level tests for ``BilevelConfig`` (no Ray).

On this branch the mechanism is declared through a ``MechanismSpace``
(``mechanism(space=..., default=...)``) and reporting through a
``ReporterConfig`` (``reporter(config)``); there is no ``wandb_cfg`` or
mechanism-template builder.
"""

from __future__ import annotations

import pytest

from core.adaptors.ray.runtime import RayRuntimeConfig
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.config import ReporterConfig


class NullReporterConfig(ReporterConfig):
    def build(self, *, label=None):
        return None


class Space:
    dimension = 2

    @classmethod
    def default(cls):
        return "space-default"


@pytest.mark.unit
def test_defaults():
    cfg = BilevelConfig()
    assert cfg.opt_class is BilevelOptimizer
    assert cfg.inner_cfg is None and cfg.outer_cfg is None
    assert cfg.outer_iters == 10 and cfg.output_dir is None
    assert cfg.world_name is None and cfg.ray_cfg is None
    assert cfg.mechanism_space is None and cfg.default_mechanism is None
    assert cfg.reporter_cfg is None
    assert cfg.env is None  # inherited OptimizerConfig fields are present


@pytest.mark.unit
def test_fluent_builders_record_settings():
    inner, outer = PPOptimizerConfig(), ESConfig()
    reporter_cfg = NullReporterConfig(project="p")
    cfg = (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(space=Space)
        .training(outer_iters=7, output_dir="out")
        .ray(device="cpu", num_cpus=2, logging_level="INFO", runtime_env={"a": 1})
        .reporter(reporter_cfg)
        .inner(inner)
        .outer(outer)
    )
    assert cfg.world_name.startswith("w_") and len(cfg.world_name) == len("w_") + 8
    assert cfg.mechanism_space is Space
    assert cfg.default_mechanism == "space-default"
    assert cfg.outer_iters == 7 and cfg.output_dir == "out"
    assert isinstance(cfg.ray_cfg, RayRuntimeConfig)
    assert cfg.ray_cfg.num_cpus == 2 and cfg.ray_cfg.logging_level == "INFO"
    assert cfg.ray_cfg.runtime_env == {"a": 1} and cfg.ray_cfg.init_kwargs == {}
    assert cfg.reporter_cfg is reporter_cfg
    assert cfg.inner_cfg is inner and cfg.outer_cfg is outer


@pytest.mark.unit
def test_explicit_default_mechanism_wins_over_space_default():
    cfg = BilevelConfig().mechanism(space=Space, default="mine")
    assert cfg.default_mechanism == "mine"


@pytest.mark.unit
def test_setters_ignore_none_and_keep_previous_values():
    inner, outer = PPOptimizerConfig(), ESConfig()
    cfg = (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(space=Space)
        .training(outer_iters=5, output_dir="out")
        .inner(inner)
        .outer(outer)
    )
    world_name = cfg.world_name

    cfg.inner(None).outer(None).world(world_name=None).mechanism(space=None)
    cfg.training(outer_iters=None)

    assert cfg.inner_cfg is inner and cfg.outer_cfg is outer
    assert cfg.world_name == world_name
    assert cfg.mechanism_space is Space and cfg.default_mechanism == "space-default"
    assert cfg.outer_iters == 5
    assert cfg.output_dir is None  # ``training`` always overwrites output_dir


@pytest.mark.unit
def test_world_name_gets_a_random_suffix():
    a = BilevelConfig().world(world_name="fishery").world_name
    b = BilevelConfig().world(world_name="fishery").world_name
    assert a.startswith("fishery_") and b.startswith("fishery_")
    assert a != b


@pytest.mark.unit
def test_config_can_be_frozen_and_copied():
    cfg = BilevelConfig().training(outer_iters=3)
    frozen = cfg.copy(copy_frozen=True)
    assert frozen.outer_iters == 3 and frozen._is_frozen
    with pytest.raises(AttributeError, match="frozen"):
        frozen.training(outer_iters=4)
    assert cfg.training(outer_iters=4).outer_iters == 4
