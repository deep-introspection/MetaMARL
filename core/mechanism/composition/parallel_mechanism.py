"""Parallel composition of mechanisms.

Given children ``(f, h, g)``, every child receives a deep copy of the same
original input and the outputs are merged::

                    -> f(x) --
                   /           |
        x --------> h(x) ------+--> merge(x, (f(x), h(x), g(x)))
                   \\           |
                    -> g(x) --

One merge function per channel decides how the outputs combine (e.g. sum the
reward shaping terms, concatenate observation augmentations). The optimizer
vector is the concatenation of the children's vectors.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.composition.chained_mechanism import _concat, _decode_children
from core.types import MultiAgentDict

MergeFn = Callable[[MultiAgentDict, tuple[MultiAgentDict, ...]], MultiAgentDict]
"""``merge(original, outputs) -> merged``; ``outputs`` follow child order."""


@dataclass(frozen=True)
class ParallelMechanism(Mechanism):
    """Parallel composition of mechanisms (see module docstring).

    Parameters
    ----------
    children : tuple[Mechanism, ...]
        Each receives the same original input.
    action_merge, reward_merge, observation_merge : MergeFn
        Combine ``(original, tuple_of_child_outputs)`` for each channel.
    """

    children: tuple[Mechanism, ...]

    action_merge: MergeFn
    reward_merge: MergeFn
    observation_merge: MergeFn

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("ParallelMechanism requires at least one child.")

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

    def decode(self, x: np.ndarray) -> ParallelMechanism:
        """Slice ``x`` by child dimension and return a copy with decoded children."""
        x = self._validate(x)
        return replace(self, children=tuple(_decode_children(self.children, x)))

    def clip(self) -> ParallelMechanism:
        """Return a copy whose children are each clipped."""
        return replace(self, children=tuple(child.clip() for child in self.children))

    # --- channels -------------------------------------------------------------

    def _fan_out(
        self, channel: str, value: MultiAgentDict, env: Any, kwargs: dict
    ) -> tuple:
        return tuple(
            getattr(child, channel)(deepcopy(value), **{**kwargs, **child.resolve(env)})
            for child in self.children
        )

    @override(Mechanism)
    def action(
        self, action_dict: MultiAgentDict, *, env: Any, **kwargs: Any
    ) -> MultiAgentDict:
        """Apply every child's ``action`` to a deep copy of ``action_dict`` and merge.

        Each child receives ``kwargs`` merged with its own bindings resolved
        against ``env``; ``action_merge(original, outputs)`` combines the
        results, with ``outputs`` in child order.
        """
        return self.action_merge(
            action_dict, self._fan_out("action", action_dict, env, kwargs)
        )

    @override(Mechanism)
    def reward(
        self, reward_dict: MultiAgentDict, *, env: Any, **kwargs: Any
    ) -> MultiAgentDict:
        """Apply every child's ``reward`` to a deep copy of ``reward_dict`` and merge.

        Each child receives ``kwargs`` (e.g. ``action_after``) merged with its
        own bindings resolved against ``env``; ``reward_merge`` combines the
        results in child order.
        """
        return self.reward_merge(
            reward_dict, self._fan_out("reward", reward_dict, env, kwargs)
        )

    @override(Mechanism)
    def observation(
        self, observation_dict: MultiAgentDict, *, env: Any, **kwargs: Any
    ) -> MultiAgentDict:
        """Apply every child's ``observation`` to a deep copy and merge.

        Each child receives ``kwargs`` merged with its own bindings resolved
        against ``env``; ``observation_merge`` combines the results in child
        order.
        """
        return self.observation_merge(
            observation_dict,
            self._fan_out("observation", observation_dict, env, kwargs),
        )
