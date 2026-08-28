"""Unit tests for the environment template classes.

``BaseEnv`` implements the template method: public ``step``/``reset`` call the
abstract ``_step``/``_reset`` hooks, apply the ``action``/``observation``/
``reward`` transforms, and publish an ``EnvStepContext`` to the World.
``RegulatedEnv`` shapes the reward with ``penalty * violation_signal`` and
``RegulatorEnv`` drives the inner optimizer for ``train_iters`` iterations.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.envs.regulator import RegulatorEnv
from core.world.context import EnvStepContext, MechanismContext, MechanismStatus


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

    env = DummyRegulator(
        world=fake_world,
        optimizer=DummyOptimizer(),
        train_iters=5,
        seeds=[100, 200],
    )
    population = np.full((3, 2), 0.5, dtype=np.float32)

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


@pytest.mark.unit
def test_regulator_env_rejects_non_positive_train_iters(fake_world):
    class DummyRegulator(RegulatorEnv):
        def _pre_reset(self, seed=None):
            pass

        def aggregate_rewards(self, ctxs):
            return 0.0

    with pytest.raises(ValueError):
        DummyRegulator(world=fake_world, optimizer=object(), train_iters=0)


class _UnitSpace:
    """Minimal mechanism space: one parameter, default candidate at 0.5."""

    dimension = 1

    @classmethod
    def default(cls):
        return SimpleNamespace(to_vector=lambda: [0.5], param_names=lambda: ["p"])

    def encode(self, m):
        return np.asarray(m.to_vector(), dtype=np.float32)

    def decode(self, x):
        return SimpleNamespace(to_vector=lambda: list(x), param_names=lambda: ["p"])


@pytest.mark.unit
def test_base_env_instantiates_mechanism_space_class_and_sets_opt_id(fake_world):
    env = _StepOnlyEnv(world=fake_world, mechanism_space=_UnitSpace)
    assert isinstance(env.m_space, _UnitSpace)
    instance = _UnitSpace()
    assert _StepOnlyEnv(world=fake_world, mechanism_space=instance).m_space is instance

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


class _Regulated(RegulatedEnv):
    def _reset(self):
        return 0

    def _step(self, action):
        return 1, 1.0, False, False, {}

    def violation_signal(self, **kwargs):
        return 0.0

    def penalty(self, **kwargs):
        return 0.0


def _published(mechanism, calls):
    def remote(**kwargs):
        calls.append(kwargs)
        return MechanismContext(
            index=0,
            env_id=None,
            seed=kwargs.get("seed"),
            status=MechanismStatus.published,
            mechanism=mechanism,
            metrics=None,
        )

    return SimpleNamespace(remote=remote)


@pytest.mark.unit
def test_regulated_env_falls_back_to_space_default_until_published(fake_world):
    fake_world.get_mechanism_by_id = SimpleNamespace(remote=lambda **kw: None)
    env = _Regulated(world=fake_world, mechanism_id=3, mechanism_space=_UnitSpace)
    assert env.m is None and env.m_ctx is None
    assert not env.published_mechanism_assigned
    assert env.mechanism.to_vector() == [0.5]
    env.reset()  # World has nothing: keep the default
    assert env.m is None and not env.published_mechanism_assigned


@pytest.mark.unit
def test_regulated_env_fetches_once_then_keeps_its_mechanism(fake_world):
    calls = []
    mechanism = SimpleNamespace(to_vector=lambda: [0.9], param_names=lambda: ["p"])
    fake_world.get_mechanism_by_id = _published(mechanism, calls)
    env = _Regulated(
        world=fake_world, mechanism_id=2, policy_seed=5, mode="eval", seed=1
    )
    env.reset()
    assert env.published_mechanism_assigned
    assert env.mechanism is mechanism and env.m_ctx.mechanism is mechanism
    assert calls == [{"mechanism_id": 2, "seed": 5, "mode": MechanismStatus.eval}]
    env.reset()
    assert len(calls) == 1  # no second fetch once a candidate is assigned
    assert fake_world.contexts[-1].payload.mechanism == 2


@pytest.mark.unit
def test_regulated_env_requires_a_mechanism_id(fake_world):
    env = _Regulated(world=fake_world, mechanism_id=None)
    with pytest.raises(RuntimeError, match="no mechanism_id"):
        env.reset()


@pytest.mark.unit
def test_regulated_env_fetch_failure_hits_missing_debug_hook(fake_world):
    """Document current behaviour: the error path is itself broken.

    ``RegulatedEnv._pre_reset`` intends to wrap a World failure in a
    ``RuntimeError`` but first calls ``self._debug_remote``, which no class in
    the hierarchy defines, so an ``AttributeError`` escapes instead.
    """

    def boom(**kwargs):
        raise ConnectionError("actor died")

    fake_world.get_mechanism_by_id = SimpleNamespace(remote=boom)
    env = _Regulated(world=fake_world, mechanism_id=0)
    with pytest.raises(AttributeError, match="_debug_remote"):
        env.reset()
