import pytest

from core.adaptors.ray.optimizer import RayOptimizer
from src.ppo.config import PPOptimizerConfig


@pytest.mark.unit
def test_ppo_config_builds_optimizer():
    cfg = PPOptimizerConfig().environment(env="CartPole-v1")
    opt = cfg.build_optimizer()
    assert isinstance(opt, RayOptimizer)
    assert opt.algo is not None


@pytest.mark.unit
def test_config_chaining():
    cfg = PPOptimizerConfig().training(lr=1e-3, gamma=0.98).resources(num_gpus=0)
    assert cfg.ray_cfg.lr == 1e-3
    assert cfg.ray_cfg.gamma == 0.98


@pytest.mark.unit
def test_freeze_blocks_mutation():
    cfg = PPOptimizerConfig().training(lr=1e-3)
    cfg.freeze()
    with pytest.raises(Exception):
        cfg.training(lr=1e-4)
