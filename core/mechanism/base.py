from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Mechanism(Protocol):
    """Semantic representation of a regulatory mechanism.

    A ``Mechanism`` encodes a point in the outer-loop search space — e.g. a
    combination of fishing quotas, fine amounts, minimum stock thresholds, and
    seasonal ban periods.  All concrete mechanisms must be convertible to and
    from a normalised float vector so that the Evolution Strategy can operate
    on a uniform Euclidean space.

    This is a structural (runtime-checkable) :class:`typing.Protocol`.  Any
    class that implements :meth:`to_vector`, :meth:`param_names`, and
    :meth:`default` satisfies the protocol without explicit subclassing.
    """

    def to_vector(self) -> list[float]:
        """Convert the mechanism to a normalised parameter vector.

        Concrete implementations must map each mechanism dimension to the
        ``[0, 1]`` interval so that the ES can treat all parameters uniformly.

        Returns
        -------
        list[float]
            Flat list of ``d`` normalised parameter values in ``[0, 1]^d``.
        """
        ...

    def param_names(self) -> list[str]:
        """Return the human-readable name of each parameter dimension.

        Returns
        -------
        list[str]
            Ordered list of parameter names corresponding to the elements
            returned by :meth:`to_vector`.
        """
        ...

    @classmethod
    def default(cls) -> "Mechanism":
        """Instantiate the canonical default mechanism.

        Concrete implementations should return a mechanism that represents a
        sensible baseline (e.g. no regulation, mid-range quotas) used when no
        mechanism has been published to the ``World`` yet.

        Returns
        -------
        Mechanism
            A default mechanism instance.
        """
        ...


@dataclass(frozen=True)
class VectorMechanism(Mechanism):
    """Lightweight mechanism backed by a raw NumPy array.

    Used as a fallback when no typed :class:`~core.mechanism.space.MechanismSpace`
    is defined.  The parameter vector is stored as-is without semantic labelling
    or normalisation guarantees.

    Parameters
    ----------
    x : np.ndarray
        Raw parameter vector.  Stored immutably (frozen dataclass).
    """

    x: np.ndarray

    def to_vector(self) -> list[float]:
        """Return the parameter vector as a flat list of ``float32`` values.

        Returns
        -------
        list[float]
            Flattened ``float32`` representation of :attr:`x`.
        """
        return np.asarray(self.x, dtype=np.float32).ravel().tolist()

    @classmethod
    def from_vector(cls, v: list[float]) -> "VectorMechanism":
        """Construct a :class:`VectorMechanism` from a list of floats.

        Parameters
        ----------
        v : list[float]
            Parameter values to store.

        Returns
        -------
        VectorMechanism
            New instance wrapping a ``float32`` NumPy array.
        """
        return cls(np.asarray(v, dtype=np.float32))
