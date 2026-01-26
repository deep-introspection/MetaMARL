import pytest

from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.envs.regulator import RegulatorEnv
from core.world.base import World


@pytest.mark.unit
def test_base_env_none_step():
    world = World()

    class TestEnv(BaseEnv):
        def _step(self, action):
            return None

        def observation(self, obs):
            return 42

        def reward(self, reward):
            return 7.0

        def action(self, action):
            return action

    env = TestEnv(world)
    obs, reward, terminated, truncated, info = env.step(1)

    assert obs == 42
    assert reward == 7.0
    assert not terminated
    assert len(world._contexts) == 1


@pytest.mark.unit
def test_regulated_env_penalty():
    class DummyRegulated(RegulatedEnv):
        def _step(self, action):
            return 1, 10.0, False, False, {}

        def violation_signal(self):
            return 2.0

        def violation_penalty(self):
            return 3.0

    dummy_world = World()
    env = DummyRegulated(dummy_world)
    obs, reward, *_ = env.step(0)

    assert reward == 10.0 - 3.0 * 2.0


@pytest.mark.unit
def test_regulator_env_triggers_optimizer():
    calls = 0

    class DummyOptimizer:
        def run(self):
            nonlocal calls
            calls += 1

    class DummyRegulator(RegulatorEnv):
        def observation(self, obs):
            return 1

        def reward(self, reward):
            return 1.0

        def action(self, action):
            return action

    dummy_world = World()
    env = DummyRegulator(dummy_world, DummyOptimizer(), iters=5)
    env.step(0)

    assert calls == 5
