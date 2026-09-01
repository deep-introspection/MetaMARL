"""Smooth threshold penalty acting on the reward channel.

When the normalized resource level ``b`` falls below ``threshold``, every agent
loses up to ``penalty_amount``:

    r*_i = r_i - penalty_amount / (1 + exp((b - threshold) / transition_width))

The penalty is a fixed regulatory rule (``dimension == 0``).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Self

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class ThresholdPenaltyMechanism(Mechanism):
    """Logistic penalty below a resource threshold (see module docstring).

    Parameters
    ----------
    threshold : float
        Resource level in ``[0, 1]`` below which the penalty applies.
    penalty_amount : float
        Maximal reward deduction, non-negative.
    transition_width : float
        Width of the logistic transition, positive.
    bindings : dict
        Must provide ``"resource_level"``: ``env -> float`` in ``[0, 1]``.
    """

    threshold: float = 0.20
    penalty_amount: float = 0.10
    transition_width: float = 0.03

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if "resource_level" not in self.bindings:
            raise ValueError(
                "ThresholdPenaltyMechanism requires a 'resource_level' binding."
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if self.penalty_amount < 0.0:
            raise ValueError("penalty_amount must be non-negative.")
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive.")

    # --- optimizer-space API -------------------------------------------------

    @property
    def dimension(self) -> int:
        """Always ``0``: the penalty is a fixed rule with no optimized parameter."""
        # Fixed regulatory rule: nothing is optimized.
        return 0

    def encode(self) -> np.ndarray:
        """Return an empty ``float32`` vector (nothing is optimized)."""
        return np.empty(0, dtype=np.float32)

    def decode(self, x: np.ndarray) -> Self:
        """Validate that ``x`` is empty and return this instance unchanged."""
        self._validate(x)
        return self

    def clip(self) -> Self:
        """Return this instance: the parameters are validated at construction."""
        return self

    def param_names(self) -> list[str]:
        """Return an empty list (nothing is optimized)."""
        return []

    def to_vector(self) -> np.ndarray:
        """Expose ``[threshold, penalty_amount]`` to agents as a ``float32`` vector.

        Both are fixed parameters; ``transition_width`` is not exposed.
        """
        return np.array([self.threshold, self.penalty_amount], dtype=np.float32)

    # --- channels -------------------------------------------------------------

    def penalty(self, resource_level: float) -> float:
        """Reward deduction for a normalized resource level."""
        z = np.clip(
            (float(resource_level) - self.threshold) / self.transition_width,
            -60.0,
            60.0,
        )
        return float(self.penalty_amount / (1.0 + np.exp(z)))

    @override(Mechanism)
    def reward(self, reward_dict: MultiAgentDict, **kwargs: Any) -> MultiAgentDict:
        """Subtract the same logistic penalty from every agent's reward.

        Consumes the ``resource_level`` binding (normalized level in
        ``[0, 1]``) from ``kwargs`` and returns rewards as Python floats.
        """
        penalty = self.penalty(kwargs["resource_level"])
        return {
            agent_id: float(reward - penalty)
            for agent_id, reward in reward_dict.items()
        }
