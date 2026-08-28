"""Builder-level tests for ``BilevelConfig`` (no Ray): reporter config and defaults.

The mechanism is declared through a template (``mechanism(mechanism=...)``)
and reporting through a ``ReporterConfig`` (``reporter(config)``).
"""

from __future__ import annotations

import pytest

from core.adaptors.ray.runtime import RayRuntimeConfig
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig
from core.reporting.config import ReporterConfig


class NullReporterConfig(ReporterConfig):
    def build(self, *, label=None):
        return None


@pytest.mark.unit
def test_defaults():
    cfg = BilevelConfig()
    assert cfg.opt_class is BilevelOptimizer
    assert cfg.inner_cfg is None and cfg.outer_cfg is None
    assert cfg.outer_iters == 10 and cfg.output_dir is None
    assert cfg.world_name is None and cfg.ray_cfg is None
    assert cfg.mechanism_template is None
    assert cfg.reporter_cfg is None
    assert cfg.env is None  # inherited OptimizerConfig fields are present


@pytest.mark.unit
def test_fluent_builders_record_settings():
    inner, outer = PPOptimizerConfig(), ESConfig()
    reporter_cfg = NullReporterConfig(project="p")
    mech = SubsidyMechanism(subsidy=0.1, cost=0.1)
    cfg = (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(mechanism=mech)
        .training(outer_iters=7, output_dir="out")
        .ray(device="cpu", num_cpus=2, logging_level="INFO", runtime_env={"a": 1})
        .reporter(reporter_cfg)
        .inner(inner)
        .outer(outer)
    )
    assert cfg.world_name.startswith("w_") and len(cfg.world_name) == len("w_") + 8
    assert cfg.mechanism_template is mech
    assert cfg.outer_iters == 7 and cfg.output_dir == "out"
    assert isinstance(cfg.ray_cfg, RayRuntimeConfig)
    assert cfg.ray_cfg.num_cpus == 2 and cfg.ray_cfg.logging_level == "INFO"
    assert cfg.ray_cfg.runtime_env == {"a": 1} and cfg.ray_cfg.init_kwargs == {}
    assert cfg.reporter_cfg is reporter_cfg
    assert cfg.inner_cfg is inner and cfg.outer_cfg is outer


@pytest.mark.unit
def test_setters_ignore_none_and_keep_previous_values():
    inner, outer = PPOptimizerConfig(), ESConfig()
    mech = SubsidyMechanism(subsidy=0.1, cost=0.1)
    cfg = (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(mechanism=mech)
        .training(outer_iters=5, output_dir="out")
        .inner(inner)
        .outer(outer)
    )
    world_name = cfg.world_name

    cfg.inner(None).outer(None).world(world_name=None).mechanism(mechanism=None)
    cfg.training(outer_iters=None)

    assert cfg.inner_cfg is inner and cfg.outer_cfg is outer
    assert cfg.world_name == world_name
    assert cfg.mechanism_template is mech
    assert cfg.outer_iters == 5
    assert cfg.output_dir is None  # ``training`` always overwrites output_dir


@pytest.mark.unit
def test_config_can_be_frozen_and_copied():
    cfg = BilevelConfig().training(outer_iters=3)
    frozen = cfg.copy(copy_frozen=True)
    assert frozen.outer_iters == 3 and frozen._is_frozen
    with pytest.raises(AttributeError, match="frozen"):
        frozen.training(outer_iters=4)
    assert cfg.training(outer_iters=4).outer_iters == 4
