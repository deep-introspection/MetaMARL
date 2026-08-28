"""Action decoding and guard paths of ``RegulatorEnv`` (no Ray).

The regulator turns the outer optimizer's action (a batch of candidate
vectors) into ``Mechanism`` objects, either through the attached mechanism
space (``decode``) or, without one, by wrapping rows in ``VectorMechanism``.
The inner optimizer is a recording stub with the call shape ``RegulatorEnv``
relies on (``run``, ``evaluate``, ``report_metrics``, ``reduce_metrics`` and a
``logger`` exposing ``peek``).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from core.envs.regulator import RegulatorEnv
from core.mechanism.base import VectorMechanism
from core.world.context import MechanismContext, MechanismStatus


class SquareSpace:
    """Decode a vector into a mechanism carrying its squared components."""

    dimension = 2

    def decode(self, x):
        return VectorMechanism(np.asarray(x, dtype=np.float32) ** 2)


class Dummy(RegulatorEnv):
    def _pre_reset(self, seed=None):
        pass

    def aggregate_rewards(self, metrics):
        return float(len(metrics))


class Inner:
    """Inner optimizer stub without a ``reset`` method."""

    def __init__(self):
        self.runs = 0
        self.evaluated = 0
        self.reported = 0
        self.logger = SimpleNamespace(peek=lambda: ["m1", "m2"])

    def run(self):
        self.runs += 1

    def evaluate(self):
        self.evaluated += 1

    def report_metrics(self):
        self.reported += 1

    def reduce_metrics(self):
        return "reduced"


def _all_vectors(mechs):
    return [m.to_vector() for m in mechs]


@pytest.mark.unit
def test_decode_through_space_accepts_list_1d_2d_and_torch(fake_world):
    env = Dummy(world=fake_world, optimizer=Inner(), mechanism_space=SquareSpace)
    for action in (
        [2.0, 3.0],
        np.array([2.0, 3.0]),
        np.array([[2.0, 3.0]]),
        torch.tensor([[2.0, 3.0]]),
    ):
        mechs = env.action(action)
        assert _all_vectors(mechs) == [[4.0, 9.0]]
    batch = env.action(np.array([[1.0, 1.0], [2.0, 2.0]]))
    assert _all_vectors(batch) == [[1.0, 1.0], [4.0, 4.0]]
    with pytest.raises(TypeError, match="Unsupported action type"):
        env.action({"not": "supported"})


@pytest.mark.unit
def test_without_space_rows_become_vector_mechanisms(fake_world):
    env = Dummy(world=fake_world, optimizer=Inner())
    for action in (
        [0.1, 0.2],
        np.array([0.1, 0.2]),
        np.array([[0.1, 0.2]]),
        torch.tensor([[0.1, 0.2]]),
    ):
        mechs = env.action(action)
        assert len(mechs) == 1 and isinstance(mechs[0], VectorMechanism)
        np.testing.assert_allclose(mechs[0].to_vector(), [0.1, 0.2], rtol=1e-6)
    # an already-built mechanism is wrapped in a singleton list
    m = VectorMechanism.from_vector([0.5])
    assert env.action(m) == [m]
    with pytest.raises(TypeError, match="no mechanism_space"):
        env.action({"not": "supported"})


@pytest.mark.unit
def test_analytic_env_passes_action_through_and_has_no_step(fake_world):
    env = Dummy(world=fake_world, optimizer=None, train_iters=0)  # allowed
    x = np.array([[0.1]])
    assert env.action(x) is x
    with pytest.raises(NotImplementedError, match="no inner optimizer"):
        env._step(x)


@pytest.mark.unit
def test_step_rejects_non_mechanism_batches(fake_world):
    env = Dummy(world=fake_world, optimizer=Inner())
    for bad in ([1, 2], np.zeros(2), "abc"):
        with pytest.raises(TypeError, match=r"list\[Mechanism\]"):
            env._step(bad)


@pytest.mark.unit
def test_reset_requires_an_observation_space_attribute(fake_world):
    """Document current behaviour: ``_reset`` reads ``observation_space``.

    ``gymnasium.Env`` only annotates ``observation_space``; nothing in the
    ``RegulatorEnv`` hierarchy assigns it, so ``reset`` on an env whose
    subclass did not set it raises ``AttributeError`` before reaching the
    ``is None`` guard.
    """
    env = Dummy(world=fake_world, optimizer=Inner())
    with pytest.raises(AttributeError, match="observation_space"):
        env.reset()


@pytest.mark.unit
def test_reset_returns_zeros_of_observation_space_or_scalar(fake_world):
    env = Dummy(world=fake_world, optimizer=Inner())
    env.observation_space = None
    obs, info = env.reset()
    assert obs == 0.0 and info == {}
    env.observation_space = spaces.Box(0, 1, (3,), np.float32)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, np.zeros(3, dtype=np.float32))
    assert obs.dtype == np.float32


@pytest.mark.unit
def test_step_trains_evaluates_flushes_and_returns_reduced_metrics(fake_world):
    inner = Inner()
    env = Dummy(
        world=fake_world,
        optimizer=inner,
        train_iters=2,
        seeds=[1, 2],
        eval_seeds=[9],
        mechanism_space=SquareSpace(),
    )
    env.observation_space = None
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([[1.0, 2.0]]))

    # inner optimizer without ``reset``: trained twice, evaluated, reported
    assert (inner.runs, inner.evaluated, inner.reported) == (2, 1, 1)
    assert reward == 2.0  # aggregate_rewards over the peeked metrics
    assert obs is None and not terminated and not truncated
    assert info == {"metrics": "reduced"}

    published = [
        c.payload
        for c in fake_world.contexts
        if isinstance(c.payload, MechanismContext)
    ]
    assert [(p.index, p.seed, p.status) for p in published] == [
        (0, 1, MechanismStatus.published),
        (0, 2, MechanismStatus.published),
    ]
    assert all(p.mechanism.to_vector() == [1.0, 4.0] for p in published)
    # eval contexts flushed before every inner run and once after training
    assert fake_world.flushed_status == [MechanismStatus.eval] * 3
    assert len(fake_world.flushed_ids) > 0


@pytest.mark.unit
def test_no_eval_seeds_skips_evaluation_and_empty_eval_list_is_none(fake_world):
    inner = Inner()
    env = Dummy(world=fake_world, optimizer=inner, train_iters=1, eval_seeds=[])
    assert env.eval_seeds is None and env.seeds == []
    env.step(np.array([[0.0, 0.0]]))
    assert inner.evaluated == 0 and inner.runs == 1
