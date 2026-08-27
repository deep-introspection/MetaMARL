"""Unit tests for the fluent Ray/RLlib optimizer configuration.

``RayOptimizerConfig`` records every RLlib builder call as a deferred
``RLlibConfigOp`` and only materializes an ``AlgorithmConfig`` inside
``build_optimizer`` (which needs a live ``World`` actor and is covered by the
integration suite). These tests check the recording and application of those
deferred operations, plus the freeze guard of the base ``OptimizerConfig``.
"""

import pytest
from ray.rllib.algorithms.ppo import PPO

from core.adaptors.ray.optimizer import RayOptimizer
from core.adaptors.ray.optimizer_config import RLlibConfigOp
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig


@pytest.mark.unit
def test_ppo_config_targets_ppo_and_ray_optimizer():
    cfg = PPOptimizerConfig()
    assert cfg.algo_class is PPO
    assert cfg.opt_class is RayOptimizer


@pytest.mark.unit
def test_config_chaining_records_deferred_ops():
    cfg = PPOptimizerConfig().training(lr=1e-3, gamma=0.98).resources(num_gpus=0)

    assert isinstance(cfg, PPOptimizerConfig)
    assert set(cfg._cfg_ops) >= {"training", "resources"}
    assert isinstance(cfg._cfg_ops["training"], RLlibConfigOp)
    assert cfg._cfg_ops["training"].kwargs == {"lr": 1e-3, "gamma": 0.98}


@pytest.mark.unit
def test_deferred_ops_apply_to_rllib_config():
    cfg = PPOptimizerConfig().training(lr=1e-3, gamma=0.98)

    rllib_cfg = PPO.get_default_config()
    for op in cfg._cfg_ops.values():
        rllib_cfg = op(rllib_cfg)

    assert rllib_cfg.lr == 1e-3
    assert rllib_cfg.gamma == 0.98


@pytest.mark.unit
def test_base_config_freeze_blocks_mutation():
    cfg = ESConfig().training(sigma=0.2)
    cfg.freeze()
    with pytest.raises(AttributeError):
        cfg.sigma = 0.3


@pytest.mark.unit
def test_ray_config_freeze_is_deferred_to_rllib():
    # On RayOptimizerConfig, ``freeze`` is an RLlib mutator: it records the
    # RLlib-side freeze instead of freezing the Python config object.
    cfg = PPOptimizerConfig().training(lr=1e-3)
    cfg.freeze()
    assert "freeze" in cfg._cfg_ops
    assert cfg._is_frozen is False
