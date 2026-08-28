"""Unit tests for the mechanism abstraction of the logging branch.

``core.mechanism.base`` exposes the ``Mechanism`` protocol and the concrete
``VectorMechanism`` value type; ``core.mechanism.space`` exposes the
``MechanismSpace`` protocol whose only concrete behaviour is ``_validate``
(shape and finiteness check of a candidate vector) and a ``default`` hook
that raises until a subclass provides one. There is no algorithm package on
this branch, so a minimal ``UnitSpace`` is defined here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.mechanism.base import Mechanism, VectorMechanism
from core.mechanism.space import MechanismSpace


class UnitSpace(MechanismSpace):
    """Two-dimensional space whose mechanisms are plain vectors."""

    dimension = 2

    def encode(self, m: Mechanism) -> np.ndarray:
        return self._validate(m.to_vector())

    def decode(self, x: np.ndarray) -> VectorMechanism:
        return VectorMechanism(self._validate(x))


@pytest.mark.unit
def test_vector_mechanism_round_trips_through_float32_lists():
    m = VectorMechanism.from_vector([0.25, 0.5])
    assert m.x.dtype == np.float32
    assert m.to_vector() == [0.25, 0.5]
    assert all(isinstance(v, float) for v in m.to_vector())
    # nested inputs are flattened
    assert VectorMechanism(np.array([[1.0], [2.0]])).to_vector() == [1.0, 2.0]


@pytest.mark.unit
def test_vector_mechanism_is_an_immutable_mechanism():
    m = VectorMechanism.from_vector([0.1])
    assert isinstance(m, Mechanism)  # explicit subclass of the protocol
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.x = np.zeros(1)


@pytest.mark.unit
def test_space_validate_returns_float32_copy_of_valid_vector():
    x = UnitSpace()._validate([1, 2])
    assert x.dtype == np.float32 and x.shape == (2,)
    np.testing.assert_allclose(x, [1.0, 2.0])


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[1.0], [[1.0, 2.0]], np.zeros((2, 1))])
def test_space_validate_rejects_wrong_shape(bad):
    with pytest.raises(ValueError, match=r"Expected shape \(2,\)"):
        UnitSpace()._validate(bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[np.nan, 0.0], [np.inf, 0.0], [0.0, -np.inf]])
def test_space_validate_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="Non-finite"):
        UnitSpace()._validate(bad)


@pytest.mark.unit
def test_space_encode_decode_use_validate():
    space = UnitSpace()
    m = space.decode([0.3, 0.7])
    assert isinstance(m, VectorMechanism)
    np.testing.assert_allclose(space.encode(m), [0.3, 0.7], rtol=1e-6)
    with pytest.raises(ValueError):
        space.decode([0.3])


@pytest.mark.unit
def test_space_default_and_unimplemented_hooks():
    with pytest.raises(NotImplementedError):
        UnitSpace.default()
    # ``clip`` and ``sample`` are protocol stubs with an ellipsis body: a
    # subclass that does not override them silently returns None.
    space = UnitSpace()
    assert space.clip(VectorMechanism.from_vector([0.0, 0.0])) is None
    assert space.sample() is None


@pytest.mark.unit
def test_space_protocol_requires_encode_and_decode():
    class Partial(MechanismSpace):
        dimension = 1

        def encode(self, m):
            return np.zeros(1)

    with pytest.raises(TypeError, match="abstract"):
        Partial()
