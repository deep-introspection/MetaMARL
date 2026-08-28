"""Tests for ``OptimizerConfig`` (copy/freeze/serialization/build) and the ``Optimizer`` graph node."""

import numpy as np
import pytest
import yaml

from core.envs.regulator import RegulatorEnv
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.optimizers.base import Optimizer
from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer


class AnalyticEnv(RegulatorEnv):
    def __init__(self, *, optimizer=None, **kwargs):
        super().__init__(optimizer=None, **kwargs)

    def _pre_reset(self, seed=None):
        pass

    def _step(self, theta):
        return None, -np.sum(np.asarray(theta) ** 2, axis=1), False, False, {}

    def aggregate_rewards(self, ctxs):
        return 0.0


@pytest.mark.unit
class TestOptimizerConfig:
    def test_environment_collects_env_config(self):
        cfg = ESConfig().environment(
            env=AnalyticEnv,
            train_iters=3,
            horizon=7,
            env_config={"a": 1},
            disable_env_checking=True,
        )
        assert cfg.env is AnalyticEnv
        assert cfg.env_config == {"train_iters": 3, "horizon": 7, "a": 1}
        assert cfg.disable_env_checking is True
        cfg._merge_env_config({"b": 2})
        assert cfg.env_config["b"] == 2

    def test_debugging_derives_seeds(self):
        cfg = ESConfig().debugging(seed=42, num_seeds=3)
        assert cfg.base_seed == 42 and len(cfg.seeds) == 3
        assert (
            cfg.seeds == ESConfig().debugging(seed=42, num_seeds=3).seeds
        )  # deterministic
        assert ESConfig().debugging().seeds == []

    def test_copy_and_freeze(self):
        cfg = ESConfig().training(sigma=0.3)
        frozen = cfg.copy(copy_frozen=True)
        assert frozen._is_frozen and not cfg._is_frozen
        with pytest.raises(AttributeError):
            frozen.sigma = 0.1
        thawed = frozen.copy(copy_frozen=False)
        thawed.sigma = 0.1
        assert thawed.sigma == 0.1 and frozen.sigma == 0.3
        same = frozen.copy()
        assert same._is_frozen
        frozen.freeze()  # idempotent

    def test_from_dict_and_yaml(self, tmp_path):
        cfg = ESConfig.from_dict({"sigma": 0.25, "unknown_key": 1})
        assert cfg.sigma == 0.25 and not hasattr(cfg, "unknown_key")
        path = tmp_path / "es.yaml"
        path.write_text(yaml.safe_dump({"sigma": 0.11, "mean_lr": 0.2}))
        cfg = ESConfig.from_yaml(str(path))
        assert cfg.sigma == 0.11 and cfg.mean_lr == 0.2

    def test_to_dict_not_implemented(self):
        with pytest.raises(NotImplementedError):
            ESConfig().to_dict()

    def test_build_optimizer_wires_world_env_and_id(self, fake_world):
        cfg = (
            ESConfig()
            .training(sigma=0.1)
            .environment(
                env=AnalyticEnv,
                env_config={"mechanism": SubsidyMechanism(subsidy=0.1, cost=0.1)},
            )
        )
        cfg.dimension = 1
        opt = cfg.build_optimizer(world=fake_world)
        assert isinstance(opt, ESOptimizer)
        assert opt.world is fake_world and opt.reporting is None
        assert opt.id == "opt_0"
        assert isinstance(opt.env, AnalyticEnv) and opt.env._opt_id == "opt_0"
        assert opt.parameter_names == ["restoration_subsidy"]
        assert opt.config._is_frozen  # the optimizer owns a frozen copy
        assert not cfg._is_frozen

    def test_build_optimizer_requires_opt_class(self):
        cfg = ESConfig()
        cfg.opt_class = None
        with pytest.raises(ValueError, match="opt_class"):
            cfg.build_optimizer()


class Leaf(Optimizer):
    def run(self):
        return {"ran": True}


@pytest.mark.unit
class TestOptimizerNode:
    def test_id_lifecycle(self):
        opt = Leaf(config=ESConfig())
        with pytest.raises(RuntimeError, match="not set"):
            _ = opt.id
        opt.set_id("a")
        assert opt.id == "a" and str(opt) == "Leaf(id=a)"
        with pytest.raises(RuntimeError, match="already set"):
            opt.set_id("b")

    def test_graph_links_and_defaults(self):
        a, b = Leaf(config=ESConfig()), Leaf(config=ESConfig())
        a.set_downstream(b)
        b.set_upstream(a)
        assert b in a._downstream and a in b._upstream
        assert a.env is None
        assert a.run() == {"ran": True}
        a.evaluate(), a.save(), a.reset(), a.stop()  # no-op defaults
        assert isinstance(Leaf.from_config(ESConfig()), Leaf)
        with pytest.raises(NotImplementedError):
            Leaf.get_default_config()
        with pytest.raises(NotImplementedError):
            Leaf.from_checkpoint("x")

    def test_env_setter_calls_hook(self):
        seen = []

        class Hooked(Leaf):
            def _on_env_init(self, env):
                seen.append(env)

        opt = Hooked(config=ESConfig())
        opt.env = "env"
        opt.env = None
        assert seen == ["env"] and opt.env is None
