"""Tests for ``ParallelMechanism`` (TODO §11)."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.mechanism.algorithms.penalty import ThresholdPenaltyMechanism
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.mechanism.composition.parallel_mechanism import ParallelMechanism
from tests.mechanism.composition.test_chained_mechanism import Affine


def additive_merge(original, outputs):
    """Sum the per-child deltas relative to the original input."""
    merged = {}
    for agent_id, base in original.items():
        base = np.asarray(base, dtype=np.float64)
        delta = sum(
            np.asarray(out[agent_id], dtype=np.float64) - base for out in outputs
        )
        merged[agent_id] = base + delta
    return merged


def last_merge(original, outputs):
    return outputs[-1]


def make(children, merge=additive_merge):
    return ParallelMechanism(
        children=children,
        action_merge=merge,
        reward_merge=merge,
        observation_merge=merge,
    )


@pytest.mark.unit
def test_requires_children():
    with pytest.raises(ValueError):
        make(())


@pytest.mark.unit
def test_every_child_sees_the_same_original_input():
    inputs = []

    class Spy(Affine):
        def _apply(self, d, **kwargs):
            inputs.append({k: np.array(v) for k, v in d.items()})
            return super()._apply(d, **kwargs)

        action = reward = observation = _apply

    m = make((Spy(scale=2.0), Spy(shift=1.0)))
    x = {"a": np.array([1.0, 2.0])}
    m.action(x, env=SimpleNamespace())
    assert len(inputs) == 2
    np.testing.assert_allclose(inputs[0]["a"], [1.0, 2.0])
    np.testing.assert_allclose(
        inputs[1]["a"], [1.0, 2.0]
    )  # not the other child's output


@pytest.mark.unit
def test_additive_merge_sums_child_deltas():
    m = make((Affine(scale=2.0), Affine(shift=1.0)))
    x = {"a": np.array([1.0, 2.0])}
    out = m.reward(x, env=SimpleNamespace())
    # x + (2x - x) + ((x + 1) - x) = 2x + 1
    np.testing.assert_allclose(out["a"], [3.0, 5.0])


@pytest.mark.unit
def test_merge_receives_original_and_ordered_outputs():
    received = {}

    def spy_merge(original, outputs):
        received["original"] = original
        received["outputs"] = outputs
        return original

    m = ParallelMechanism(
        children=(Affine(scale=2.0), Affine(scale=3.0)),
        action_merge=spy_merge,
        reward_merge=spy_merge,
        observation_merge=spy_merge,
    )
    x = {"a": np.array([1.0])}
    m.observation(x, env=SimpleNamespace())
    assert received["original"] is x
    assert len(received["outputs"]) == 2
    np.testing.assert_allclose(received["outputs"][0]["a"], [2.0])
    np.testing.assert_allclose(received["outputs"][1]["a"], [3.0])


@pytest.mark.unit
def test_deep_copies_prevent_cross_child_mutation():
    class Mutator(Affine):
        def _apply(self, d, **kwargs):
            for v in d.values():
                np.asarray(v)[...] = -99.0
            return d

        action = reward = observation = _apply

    m = make((Mutator(), Affine(scale=1.0)), merge=last_merge)
    x = {"a": np.array([1.0, 2.0])}
    out = m.action(x, env=SimpleNamespace())
    np.testing.assert_allclose(x["a"], [1.0, 2.0])
    np.testing.assert_allclose(out["a"], [1.0, 2.0])


@pytest.mark.unit
def test_vector_api_and_bindings():
    binding = {"resource_level": lambda env: env.level}
    m = make(
        (
            ThresholdPenaltyMechanism(bindings=binding, penalty_amount=0.1),
            SubsidyMechanism(subsidy=0.2, cost=0.0),
        )
    )
    assert m.dimension == 1
    assert m.param_names() == ["1:SubsidyMechanism.restoration_subsidy"]
    np.testing.assert_allclose(m.encode(), [0.4])
    decoded = m.decode(np.array([1.0]))
    assert decoded.children[1].subsidy == pytest.approx(0.5)
    assert m.clip().children[1].subsidy == pytest.approx(0.2)

    env = SimpleNamespace(level=0.0)  # full penalty
    out = m.reward({"a": 1.0}, env=env, action_after={"a": np.array([0.0, 1.0])})
    assert out["a"] == pytest.approx(1.0 - 0.1 + 0.2, abs=1e-3)
