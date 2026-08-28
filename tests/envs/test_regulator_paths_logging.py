"""Reset and inner-loop reporting paths of ``RegulatorEnv`` (no Ray).

Action decoding is covered by ``test_regulator_paths.py``. Here the inner
optimizer is a recording stub with the call shape ``RegulatorEnv`` relies on
(``run``, ``evaluate``, ``report_metrics``, ``reduce_metrics`` and a ``logger``
exposing ``peek``): one outer step must report the inner metrics, hand the
peeked schema to ``aggregate_rewards`` and return the reduced schema in
``info``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.envs.regulator import RegulatorEnv
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.world.context import MechanismContext, MechanismStatus


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
def test_step_trains_evaluates_reports_flushes_and_returns_reduced_metrics(
    fake_world,
):
    inner = Inner()
    env = Dummy(
        world=fake_world,
        optimizer=inner,
        train_iters=2,
        seeds=[1, 2],
        eval_seeds=[9],
        mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1),
    )
    env.observation_space = None
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([[0.3]]))

    # inner optimizer without ``reset``: trained twice, evaluated, reported once
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
    # candidates are decoded through the template (subsidy normalized by 0.5)
    assert all(p.mechanism.subsidy == pytest.approx(0.15) for p in published)
    # eval contexts flushed before every inner run and once after training
    assert fake_world.flushed_status == [MechanismStatus.eval] * 3
    assert len(fake_world.flushed_ids) > 0


@pytest.mark.unit
def test_inner_without_metric_logger_hands_none_to_aggregation(fake_world):
    class NoneAggregator(Dummy):
        def aggregate_rewards(self, metrics):
            assert metrics is None
            return 0.0

    inner = Inner()
    inner.logger = None
    env = NoneAggregator(
        world=fake_world,
        optimizer=inner,
        train_iters=1,
        mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1),
    )
    _, reward, *_ = env.step(np.array([[0.3]]))
    assert reward == 0.0 and inner.reported == 1


@pytest.mark.unit
def test_no_eval_seeds_skips_evaluation_and_empty_eval_list_is_none(fake_world):
    inner = Inner()
    env = Dummy(
        world=fake_world,
        optimizer=inner,
        train_iters=1,
        eval_seeds=[],
        mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1),
    )
    assert env.eval_seeds is None and env.seeds == []
    env.step(np.array([[0.0]]))
    assert inner.evaluated == 0 and inner.runs == 1
