"""Tests for ``ChainedMechanism`` (TODO §10)."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest

from core.mechanism.algorithms.penalty import ThresholdPenaltyMechanism
from core.mechanism.algorithms.quota import QuotaMechanism
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.mechanism.base import Mechanism
from core.mechanism.composition.chained_mechanism import ChainedMechanism


@dataclass(frozen=True)
class Affine(Mechanism):
    """Test mechanism: x -> scale * x + shift on every channel; ``scale`` optimized."""

    scale: float = 1.0
    shift: float = 0.0
    bindings: dict = None

    @property
    def dimension(self) -> int:
        return 1

    def encode(self):
        return np.array([self.scale], dtype=np.float32)

    def decode(self, x) -> Self:
        return replace(self, scale=float(self._validate(x)[0]))

    def clip(self) -> Self:
        return replace(self, scale=float(np.clip(self.scale, 0.0, 1.0)))

    def param_names(self):
        return ["scale"]

    def to_vector(self):
        return np.array([self.scale, self.shift], dtype=np.float32)

    def _apply(self, d, **kwargs):
        return {k: self.scale * np.asarray(v) + self.shift for k, v in d.items()}

    action = observation = reward = _apply


@pytest.mark.unit
def test_requires_children():
    with pytest.raises(ValueError):
        ChainedMechanism(children=())


@pytest.mark.unit
def test_order_is_child_tuple_order_on_all_channels():
    chain = ChainedMechanism(children=(Affine(scale=2.0), Affine(shift=1.0)))
    env = SimpleNamespace()
    x = {"a": np.array([1.0, 2.0])}
    for channel in ("action", "reward", "observation"):
        out = getattr(chain, channel)(x, env=env)
        np.testing.assert_allclose(
            out["a"], 2 * np.array([1.0, 2.0]) + 1
        )  # 2x + 1, not 2(x+1)


@pytest.mark.unit
def test_vector_concatenation_and_slicing():
    chain = ChainedMechanism(children=(Affine(scale=0.2), Affine(scale=0.7, shift=5.0)))
    assert chain.dimension == 2
    assert chain.param_names() == ["0:Affine.scale", "1:Affine.scale"]
    np.testing.assert_allclose(chain.encode(), [0.2, 0.7])
    np.testing.assert_allclose(chain.to_vector(), [0.2, 0.0, 0.7, 5.0])

    decoded = chain.decode(np.array([0.9, 0.1]))
    assert decoded.children[0].scale == pytest.approx(0.9)
    assert decoded.children[1].scale == pytest.approx(0.1)
    assert decoded.children[1].shift == 5.0
    assert chain.children[0].scale == 0.2  # original untouched


@pytest.mark.unit
def test_zero_dimension_children_do_not_break_slicing():
    binding = {"resource_level": lambda env: 1.0}
    chain = ChainedMechanism(
        children=(
            ThresholdPenaltyMechanism(bindings=binding),
            QuotaMechanism(fixed_quota=0.5, bindings=binding),
            SubsidyMechanism(subsidy=0.1, cost=0.1),
        )
    )
    assert chain.dimension == 2
    assert chain.param_names() == [
        "1:QuotaMechanism.fixed_quota",
        "2:SubsidyMechanism.restoration_subsidy",
    ]
    decoded = chain.decode(np.array([0.3, 1.0]))
    assert decoded.children[1].fixed_quota == pytest.approx(0.3)
    assert decoded.children[2].subsidy == pytest.approx(0.5)
    np.testing.assert_allclose(decoded.encode(), [0.3, 1.0], atol=1e-6)


@pytest.mark.unit
def test_clip_propagates_to_children():
    chain = ChainedMechanism(children=(Affine(scale=3.0), Affine(scale=-1.0)))
    clipped = chain.clip()
    assert [c.scale for c in clipped.children] == [1.0, 0.0]


@pytest.mark.unit
def test_each_child_resolves_its_own_bindings():
    seen = {}
    binding_a = {"resource_level": lambda env: seen.setdefault("a", env.level_a)}
    binding_b = {"resource_level": lambda env: seen.setdefault("b", env.level_b)}
    chain = ChainedMechanism(
        children=(
            QuotaMechanism(fixed_quota=0.5, bindings=binding_a),
            ThresholdPenaltyMechanism(bindings=binding_b),
        )
    )
    env = SimpleNamespace(level_a=0.9, level_b=0.1)
    chain.action({"a": np.array([0.5, 0.5])}, env=env)
    chain.reward({"a": 1.0}, env=env)
    assert seen == {"a": 0.9, "b": 0.1}


@pytest.mark.unit
def test_full_fishery_stack_reward_and_action():
    binding = {"resource_level": lambda env: env.level}
    chain = ChainedMechanism(
        children=(
            QuotaMechanism(fixed_quota=0.5, bindings=binding),
            SubsidyMechanism(subsidy=0.2, cost=0.1, action_component=1),
        )
    )
    env = SimpleNamespace(level=0.95)
    actions = chain.action({"a": np.array([0.4, 0.5])}, env=env)
    np.testing.assert_allclose(actions["a"], [0.4, 0.5], atol=1e-4)
    rewards = chain.reward({"a": 1.0}, env=env, action_after=actions)
    assert rewards["a"] == pytest.approx(1.0 + 0.2 * 0.5 - 0.1 * 0.25)
