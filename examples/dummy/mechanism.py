from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class DummyMechanism(Mechanism):
    """Minimal no-op mechanism used for testing and framework validation.

    Carries a single scalar ``value`` that is not consumed by any
    environment logic.  Useful when a mechanism object is required by the
    bilevel API but the experiment does not need real regulatory parameters
    (e.g., the CartPole toy example).

    Parameters
    ----------
    value : float, optional
        Nominal mechanism value.  Has no effect on environment dynamics.
        Default is ``0.0``.
    """

    value: float = 0.0

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        """Encode the mechanism as a zero-length vector.

        The dummy mechanism has no optimizable parameters, so the ES outer
        loop operates in a trivial 0-D space.

        Returns
        -------
        np.ndarray
            Empty float32 array of shape ``(0,)``.
        """
        return np.array([], dtype=np.float32)


class DummyMechanismSpace(MechanismSpace):
    """Mechanism space stub wrapping a single scalar parameter.

    Satisfies the :class:`~core.mechanism.space.MechanismSpace` interface
    while keeping the ES search space trivially small (dimension = 1).
    Intended for use with toy environments such as CartPole where the
    outer loop must exist structurally but does not need to optimize real
    regulatory parameters.
    """

    def __init__(self):
        super().__init__()
        self.dimension = 1
        self.full_dimension = 0

    def default(self) -> DummyMechanism:
        """Return the default mechanism with ``value = 0.0``.

        Returns
        -------
        DummyMechanism
            Default no-op mechanism instance.
        """
        return DummyMechanism(0.0)

    def encode(self, m: DummyMechanism) -> NDArray[np.float32]:
        """Encode a :class:`DummyMechanism` as a 1-D float32 vector.

        Parameters
        ----------
        m : DummyMechanism
            Mechanism instance to encode.

        Returns
        -------
        NDArray[np.float32]
            Array of shape ``(1,)`` containing ``m.value``.
        """
        return np.array([m.value], dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        """Decode a 1-D float32 vector into a :class:`DummyMechanism`.

        Parameters
        ----------
        x : NDArray[np.float32]
            Encoded mechanism vector of shape ``(1,)``.

        Returns
        -------
        Mechanism
            Decoded :class:`DummyMechanism` instance.
        """
        x = self._validate(x)
        return DummyMechanism(float(x[0]))

    def clip(self, m: DummyMechanism) -> DummyMechanism:
        """Return a clipped copy of the mechanism (identity for the dummy space).

        Parameters
        ----------
        m : DummyMechanism
            Mechanism to clip.

        Returns
        -------
        DummyMechanism
            Clipped mechanism (value unchanged, since no bounds apply).
        """
        return DummyMechanism(float(m.value))

    def from_dict(self, cfg: dict) -> DummyMechanism:
        """Construct a :class:`DummyMechanism` from a configuration dictionary.

        Parameters
        ----------
        cfg : dict
            Configuration mapping.  The optional key ``"value"`` is used;
            defaults to ``0.0`` if absent.

        Returns
        -------
        DummyMechanism
            Mechanism instance built from ``cfg``.
        """
        return DummyMechanism(float(cfg.get("value", 0.0)))
