"""Sequential composition of mechanisms.

For children ``(f, h, g)`` every channel is the functional composition
``g(h(f(x)))``: each child receives the previous child's output. Each child
resolves its own bindings against the environment. The optimizer vector is the
concatenation of the children's vectors, sliced back on ``decode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class ChainedMechanism(Mechanism):
    """Functional composition of mechanisms (see module docstring).

    Parameters
    ----------
    children : tuple[Mechanism, ...]
        Applied in tuple order on every channel.
    """

    children: tuple[Mechanism, ...]

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("ChainedMechanism requires at least one child.")

    # --- optimizer-space API -------------------------------------------------

    @property
    def dimension(self) -> int:
        """Sum of the children's dimensions."""
        return sum(child.dimension for child in self.children)

    def param_names(self) -> list[str]:
        """Children's parameter names prefixed by ``"<index>:<ClassName>."``."""
        names: list[str] = []
        for index, child in enumerate(self.children):
            prefix = f"{index}:{type(child).__name__}"
            names.extend(f"{prefix}.{name}" for name in child.param_names())
        return names

    def encode(self) -> np.ndarray:
        """Concatenate the children's encodings in tuple order."""
        return _concat(child.encode() for child in self.children)

    def to_vector(self) -> np.ndarray:
        """Concatenate the children's agent-facing vectors in tuple order."""
        return _concat(child.to_vector() for child in self.children)

    def decode(self, x: np.ndarray) -> ChainedMechanism:
        """Slice ``x`` by child dimension and return a copy with decoded children."""
        x = self._validate(x)
        return replace(self, children=tuple(_decode_children(self.children, x)))

    def clip(self) -> ChainedMechanism:
        """Return a copy whose children are each clipped."""
        return replace(self, children=tuple(child.clip() for child in self.children))

    # --- channels -------------------------------------------------------------

    @override(Mechanism)
    def reward(
        self, reward_dict: MultiAgentDict, *, env: Any, **kwargs: Any
    ) -> MultiAgentDict:
        """Pass the rewards through each child's ``reward`` in tuple order.

        Each child receives ``kwargs`` (e.g. ``action_after``) merged with its
        own bindings resolved against ``env``; the composite's own bindings are
        not used.
        """
        for child in self.children:
            reward_dict = child.reward(reward_dict, **{**kwargs, **child.resolve(env)})
        return reward_dict

    @override(Mechanism)
    def action(
        self, action_dict: MultiAgentDict, *, env: Any, **kwargs: Any
    ) -> MultiAgentDict:
        """Pass the actions through each child's ``action`` in tuple order.

        Each child receives ``kwargs`` merged with its own bindings resolved
        against ``env``.
        """
        for child in self.children:
            action_dict = child.action(action_dict, **{**kwargs, **child.resolve(env)})
        return action_dict

    @override(Mechanism)
    def observation(
        self, observation_dict: MultiAgentDict, *, env: Any, **kwargs: Any
    ) -> MultiAgentDict:
        """Pass the observations through each child's ``observation`` in tuple order.

        Each child receives ``kwargs`` merged with its own bindings resolved
        against ``env``, so observation augmentations accumulate.
        """
        for child in self.children:
            observation_dict = child.observation(
                observation_dict, **{**kwargs, **child.resolve(env)}
            )
        return observation_dict


def _concat(vectors) -> np.ndarray:
    parts = [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors]
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def _decode_children(children: tuple[Mechanism, ...], x: np.ndarray) -> list[Mechanism]:
    """Slice ``x`` by child dimension and decode each child."""
    decoded: list[Mechanism] = []
    start = 0
    for child in children:
        stop = start + child.dimension
        decoded.append(child.decode(x[start:stop]))
        start = stop
    return decoded
