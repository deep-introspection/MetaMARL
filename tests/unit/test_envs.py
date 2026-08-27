"""Unit tests for the environment template classes.

``BaseEnv`` implements the template method: public ``step``/``reset`` call the
abstract ``_step``/``_reset`` hooks, apply the ``action``/``observation``/
``reward`` transforms, and publish an ``EnvStepContext`` to the World.
``RegulatedEnv`` shapes the reward with ``penalty * violation_signal`` and
``RegulatorEnv`` drives the inner optimizer for ``train_iters`` iterations.
"""

import numpy as np
import pytest

from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.envs.regulator import RegulatorEnv
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
def test_regulated_env_penalty_shapes_reward(fake_world):
    class DummyRegulated(RegulatedEnv):
        def _pre_reset(self, seed=None):
            pass

        def _reset(self):
            return 0

        def _step(self, action):
            return 1, 10.0, False, False, {}

        def violation_signal(self, **kwargs):
            return 2.0

        def penalty(self, **kwargs):
            return 3.0

    env = DummyRegulated(world=fake_world, mechanism_id=0)
    obs, reward, *_ = env.step(0)

    assert obs == 1
    assert reward == 10.0 - 3.0 * 2.0


@pytest.mark.unit
def test_regulator_env_trains_inner_optimizer_and_publishes_mechanisms(fake_world):
    calls = {"run": 0, "reset": 0}

    class DummyOptimizer:
        def run(self):
            calls["run"] += 1

        def reset(self):
            calls["reset"] += 1

    class DummyRegulator(RegulatorEnv):
        def _pre_reset(self, seed=None):
            pass

        def aggregate_rewards(self, ctxs):
            return 1.0

    env = DummyRegulator(
        world=fake_world,
        optimizer=DummyOptimizer(),
        train_iters=5,
        seeds=[100, 200],
    )
    population = np.full((3, 2), 0.5, dtype=np.float32)

    _, reward, *_ = env.step(population)

    assert reward == 1.0
    assert calls == {"run": 5, "reset": 1}

    mechanism_ctxs = [
        c.payload
        for c in fake_world.contexts
        if isinstance(c.payload, MechanismContext)
    ]
    # one MechanismContext per (candidate, seed)
    assert len(mechanism_ctxs) == 3 * 2
    assert sorted({c.index for c in mechanism_ctxs}) == [0, 1, 2]
    assert sorted({c.seed for c in mechanism_ctxs}) == [100, 200]


@pytest.mark.unit
def test_regulator_env_rejects_non_positive_train_iters(fake_world):
    class DummyRegulator(RegulatorEnv):
        def _pre_reset(self, seed=None):
            pass

        def aggregate_rewards(self, ctxs):
            return 0.0

    with pytest.raises(ValueError):
        DummyRegulator(world=fake_world, optimizer=object(), train_iters=0)
