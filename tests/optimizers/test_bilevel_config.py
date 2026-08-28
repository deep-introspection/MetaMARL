"""Builder-level tests for ``BilevelConfig`` (no Ray)."""

import pytest

from core.adaptors.ray.runtime import RayRuntimeConfig
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig


@pytest.mark.unit
def test_fluent_builders_record_settings():
    inner, outer = PPOptimizerConfig(), ESConfig()
    mech = SubsidyMechanism(subsidy=0.1, cost=0.1)
    cfg = (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(mechanism=mech)
        .training(outer_iters=7, output_dir="out")
        .ray(device="cpu", num_cpus=2)
        .inner(inner)
        .outer(outer)
    )
    assert cfg.world_name.startswith("w_") and len(cfg.world_name) == len("w_") + 8
    assert cfg.mechanism_template is mech
    assert cfg.outer_iters == 7 and cfg.output_dir == "out"
    assert isinstance(cfg.ray_cfg, RayRuntimeConfig) and cfg.ray_cfg.num_cpus == 2
    assert cfg.inner_cfg is inner and cfg.outer_cfg is outer
    assert cfg.reporter_cfg is None  # see test_bilevel_config_logging.py


@pytest.mark.unit
def test_mechanism_must_be_a_mechanism():
    with pytest.raises(TypeError, match="Mechanism instance"):
        BilevelConfig().mechanism(mechanism=object())


@pytest.mark.unit
def test_build_requires_mechanism_and_both_levels():
    with pytest.raises(ValueError, match="mechanism"):
        BilevelConfig().inner(PPOptimizerConfig()).outer(ESConfig()).build_optimizer()
    with pytest.raises(ValueError, match="inner"):
        BilevelConfig().mechanism(
            mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1)
        ).build_optimizer()
