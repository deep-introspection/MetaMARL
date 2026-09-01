"""Abstract mechanism interface.

A mechanism is a regulatory intervention on the agent/environment loop. It
acts through three channels, each an optional transform applied by the
environment at a fixed point of the step:

- ``action``: ``a* = M^A(s, a)`` -- e.g. a quota capping harvest requests;
- ``reward``: ``r* = M^R(r, s, a*, s')`` -- e.g. a subsidy or a penalty;
- ``observation``: ``o* = M^O(s, o)`` -- e.g. exposing peers' past actions.

A mechanism is also a point in an optimizer space: ``encode()`` maps its free
parameters to a vector in ``[0, 1]^dimension`` and ``decode(x)`` returns the
same mechanism structure parameterized by ``x``. Fixed mechanisms have
``dimension == 0``. Composite mechanisms (see ``core.mechanism.composition``)
concatenate their children's vectors.

Mechanisms are immutable dataclasses; per-step context they need from the
environment (resource level, previous actions, ...) is injected through
``bindings``: callables ``env -> value`` resolved by :meth:`resolve` and passed
as keyword arguments to the channel methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Self

import numpy as np

from core.types import MultiAgentDict


class Mechanism(ABC):
    """Semantic representation of a regulatory mechanism.

    Subclasses must implement the optimizer-space API (``dimension``,
    ``encode``, ``decode``, ``clip``, ``param_names``, ``to_vector``) and may
    override any of the three channel transforms, which default to identity.
    """

    bindings: dict[str, Callable[[Any], Any]]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of optimized parameters (size of :meth:`encode`)."""
        ...

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Normalized optimizer representation, shape ``(dimension,)`` in ``[0, 1]``."""
        ...

    @abstractmethod
    def decode(self, x: np.ndarray) -> Self:
        """Return the same mechanism structure parameterized by ``x``.

        For composite mechanisms, decoding propagates recursively to children.
        """
        ...

    @abstractmethod
    def clip(self) -> Self:
        """Return a copy whose parameters are clipped to their valid ranges."""
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Names corresponding exactly to the entries of :meth:`encode`."""
        ...

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """Full semantic representation exposed to agents (normalized floats).

        Unlike :meth:`encode`, this may include fixed (non-optimized)
        parameters; the environment appends it to every agent observation.
        """
        ...

    def _validate(self, x: np.ndarray) -> np.ndarray:
        """Check that ``x`` is a finite vector of shape ``(dimension,)``."""
        x = np.asarray(x, dtype=np.float32)

        if x.shape != (self.dimension,):
            raise ValueError(f"Expected shape ({self.dimension},), got {x.shape}")

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in vector: {x}")

        return x

    def resolve(self, env: Any) -> dict[str, Any]:
        """Evaluate every binding against ``env`` and return ``{name: value}``."""
        bindings = getattr(self, "bindings", None) or {}
        return {name: binding(env) for name, binding in bindings.items()}

    def action(self, action_dict: MultiAgentDict, **kwargs: Any) -> MultiAgentDict:
        """Transform agent actions (identity by default)."""
        return action_dict

    def observation(
        self, observation_dict: MultiAgentDict, **kwargs: Any
    ) -> MultiAgentDict:
        """Transform agent observations (identity by default)."""
        return observation_dict

    def reward(self, reward_dict: MultiAgentDict, **kwargs: Any) -> MultiAgentDict:
        """Transform agent rewards (identity by default)."""
        return reward_dict
