from abc import abstractmethod
from typing import Protocol

import numpy as np

from core.mechanism.base import Mechanism


# TODO why not have these methods be abstract ?
class MechanismSpace(Protocol):
    """Geometry and constraints over a mechanism manifold.

    A ``MechanismSpace`` defines the bijection between the semantic
    :class:`~core.mechanism.base.Mechanism` domain (typed, interpretable
    parameters such as quotas and fines) and the normalised Euclidean search
    space used by the outer-loop Evolution Strategy.

    Each concrete space specifies:

    - ``dimension`` — the number of searchable parameters.
    - :meth:`encode` / :meth:`decode` — round-trip between mechanism objects
      and ``[0, 1]^d`` vectors.
    - :meth:`clip` — project an out-of-bounds mechanism back into feasibility.
    - :meth:`sample` — draw a uniformly random mechanism for initialisation.

    This is a structural :class:`typing.Protocol`; implementors need not
    explicitly subclass it.

    Attributes
    ----------
    dimension : int
        Number of mechanism parameters (dimensionality of the search space).
    """

    dimension: int

    def _validate(self, x: np.ndarray) -> np.ndarray:
        """Validate that a parameter vector is finite and correctly shaped.

        Parameters
        ----------
        x : np.ndarray
            Candidate parameter vector to validate.

        Returns
        -------
        np.ndarray
            The input cast to ``float32`` if valid.

        Raises
        ------
        ValueError
            If the shape does not match ``(dimension,)`` or if any value is
            non-finite (``NaN`` or ``inf``).
        """
        x = np.asarray(x, dtype=np.float32)

        if x.shape != (self.dimension,):
            raise ValueError(f"Expected shape ({self.dimension},), got {x.shape}")

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in vector: {x}")

        return x

    @classmethod
    def default(cls) -> "Mechanism":
        """Return the canonical default mechanism for this space.

        Concrete implementations should return a mechanism that represents a
        sensible regulatory baseline (e.g. zero fines, maximum quota) used
        when no mechanism has been published to the ``World``.

        Returns
        -------
        Mechanism
            Default mechanism instance.

        Raises
        ------
        NotImplementedError
            If the concrete class does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, m: Mechanism) -> np.ndarray:
        """Encode a typed mechanism into a normalised parameter vector.

        Concrete implementations must map each mechanism parameter to ``[0, 1]``
        so that the ES can treat all dimensions uniformly.

        Parameters
        ----------
        m : Mechanism
            Typed mechanism object to encode.

        Returns
        -------
        np.ndarray
            Normalised vector of shape ``(dimension,)`` with values in
            ``[0, 1]``.
        """
        ...

    @abstractmethod
    def decode(self, x: np.ndarray) -> Mechanism:
        """Decode a normalised parameter vector into a typed mechanism.

        Inverse of :meth:`encode`.  Concrete implementations must map each
        ``[0, 1]``-normalised value back to its physical scale and construct
        the appropriate :class:`~core.mechanism.base.Mechanism` object.

        Parameters
        ----------
        x : np.ndarray
            Normalised vector of shape ``(dimension,)``.

        Returns
        -------
        Mechanism
            Fully typed mechanism object.
        """
        ...

    def clip(self, m: Mechanism) -> Mechanism:
        """Project a mechanism back into the feasible region of this space.

        Default implementation is a no-op.  Concrete spaces that have hard
        constraints (e.g. quota must not exceed carrying capacity) should
        override this to clamp out-of-bounds parameter values.

        Parameters
        ----------
        m : Mechanism
            Mechanism to project.

        Returns
        -------
        Mechanism
            Feasible mechanism (may be the same object if already in bounds).
        """
        ...

    def sample(self) -> Mechanism:
        """Sample a uniformly random mechanism from this space.

        Used during ES initialisation to populate the first generation of
        mechanism candidates.  Default implementation is a no-op placeholder;
        concrete spaces should draw each parameter independently from
        ``Uniform(0, 1)`` (or the appropriate prior) and call :meth:`decode`.

        Returns
        -------
        Mechanism
            A randomly sampled mechanism.
        """
        ...
