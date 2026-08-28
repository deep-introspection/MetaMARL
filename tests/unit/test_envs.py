"""Unit tests for the environment template classes.

``BaseEnv`` implements the template method: public ``step``/``reset`` call the
abstract ``_step``/``_reset`` hooks, apply the ``action``/``observation``/
``reward`` transforms, and publish an ``EnvStepContext`` to the World.
``RegulatorEnv`` drives the inner optimizer for ``train_iters`` iterations. The
multi-agent regulated env has its own suite in ``tests/envs/``.
"""

import numpy as np
import pytest

from core.envs.base import BaseEnv
from core.envs.regulator import RegulatorEnv
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.world.context import EnvStepContext, MechanismContext


class _StepOnlyEnv(BaseEnv):
    def _pre_reset(self, seed=None):
        pass

    def _reset(self):
        return 0

    def _step(self, action):
        return action, float(action), False, False, {}


@pytest.mark.unit
def test_base_env_step_applies_transforms_and_publishes(fake_world):
    class TestEnv(_StepOnlyEnv):
        def observation(self, obs):
            return 42

        def reward(self, reward):
            return 7.0

        def action(self, action):
            return action * 2

    env = TestEnv(world=fake_world)
    obs, reward, terminated, truncated, info = env.step(1)

    assert obs == 42
    assert reward == 7.0
    assert not terminated and not truncated
    assert info == {}
    assert env._t == 1

    assert len(fake_world.contexts) == 1
    payload = fake_world.contexts[0].payload
    assert isinstance(payload, EnvStepContext)
    assert payload.observation == 42
    assert payload.reward == 7.0
    assert payload.action == 1  # raw action is published, not the transformed one


@pytest.mark.unit
def test_base_env_reset_publishes_and_rewinds_time(fake_world):
    env = _StepOnlyEnv(world=fake_world)
    env.step(1)
    env.step(1)
    assert env._t == 2

    obs, info = env.reset()

    assert obs == 0
    assert info == {}
    assert env._t == 0
    assert len(fake_world.contexts) == 3
    assert fake_world.contexts[-1].payload.reward == 0.0
    assert fake_world.contexts[-1].payload.action is None


@pytest.mark.unit
def test_regulator_env_trains_inner_optimizer_and_publishes_mechanisms(fake_world):
    calls = {"run": 0, "reset": 0}

    class DummyLogger:
        def peek(self):
            return "peeked"

        def reduce(self):
            return "reduced"

    class DummyOptimizer:
        logger = DummyLogger()

        def run(self):
            calls["run"] += 1

        def reset(self):
            calls["reset"] += 1

        def report_metrics(self):
            calls["report"] = calls.get("report", 0) + 1

        def reduce_metrics(self):
            return self.logger.reduce()

    class DummyRegulator(RegulatorEnv):
        def _pre_reset(self, seed=None):
            pass

        def aggregate_rewards(self, metrics):
            assert metrics == "peeked"  # the inner optimizer's peeked metrics
            return 1.0

    template = SubsidyMechanism(subsidy=0.1, cost=0.1)
    env = DummyRegulator(
        world=fake_world,
        optimizer=DummyOptimizer(),
        train_iters=5,
        seeds=[100, 200],
        mechanism=template,
    )
    population = np.array([[0.2], [0.5], [0.8]], dtype=np.float32)

    _, reward, _, _, info = env.step(population)

    assert reward == 1.0
    assert calls == {"run": 5, "reset": 1, "report": 1}
    assert info == {"metrics": "reduced"}

    mechanism_ctxs = [
        c.payload
        for c in fake_world.contexts
        if isinstance(c.payload, MechanismContext)
    ]
    # one MechanismContext per (candidate, seed)
    assert len(mechanism_ctxs) == 3 * 2
    assert sorted({c.index for c in mechanism_ctxs}) == [0, 1, 2]
    assert sorted({c.seed for c in mechanism_ctxs}) == [100, 200]
    # candidates are decoded through the template
    assert all(isinstance(c.mechanism, SubsidyMechanism) for c in mechanism_ctxs)
    assert sorted({round(c.mechanism.subsidy, 3) for c in mechanism_ctxs}) == [
        0.1,
        0.25,
        0.4,
    ]


@pytest.mark.unit
def test_regulator_env_rejects_non_positive_train_iters(fake_world):
    class DummyRegulator(RegulatorEnv):
        def _pre_reset(self, seed=None):
            pass

        def aggregate_rewards(self, ctxs):
            return 0.0

    with pytest.raises(ValueError):
        DummyRegulator(world=fake_world, optimizer=object(), train_iters=0)


@pytest.mark.unit
def test_base_env_set_opt_id_stamps_published_contexts(fake_world):
    env = _StepOnlyEnv(world=fake_world)
    env.set_opt_id("opt_7")
    env.step(1)
    assert fake_world.contexts[-1].opt_id == "opt_7"
    assert fake_world.contexts[-1].env == "_StepOnlyEnv"


@pytest.mark.unit
def test_base_env_reset_ignores_a_different_seed(fake_world):
    env = _StepOnlyEnv(world=fake_world, seed=11)
    first_draw = env.rng.random()
    rng_state = env.rng.bit_generator.state
    env.reset(seed=99)
    assert env.seed == 11
    assert env.rng.bit_generator.state == rng_state  # rng not re-seeded
    assert env.rng.random() != first_draw
    assert fake_world.contexts[-1].payload.seed == 11
