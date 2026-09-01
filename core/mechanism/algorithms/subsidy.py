"""Restoration subsidy acting on the reward channel.

For agent ``i`` with restoration effort ``e_i`` (one action component in
``[0, 1]``), subsidy rate ``sigma`` and quadratic effort cost ``c``:

    r*_i = r_i + sigma * e_i - c * e_i**2

The subsidy rate is the optimized parameter, normalized by ``MAX_SUBSIDY``; the
cost is a fixed benchmark parameter. Ecological effects of restoration belong
to the benchmark transition, not to this mechanism.
"""

from dataclasses import dataclass, replace
from typing import Any, Self

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict

MAX_SUBSIDY = 0.5
"""Upper bound of the subsidy rate; ``encode()`` divides by it."""


@dataclass(frozen=True)
class SubsidyMechanism(Mechanism):
    """Reward-channel subsidy on restoration effort (see module docstring).

    Parameters
    ----------
    subsidy : float
        Subsidy rate in ``[0, MAX_SUBSIDY]``. Optimized (``dimension == 1``).
    cost : float
        Quadratic effort cost in ``[0, 1]``. Fixed.
    action_component : int
        Index of the action component holding the restoration effort.

    The reward transform reads the delivered actions from the keyword argument
    ``action_after`` supplied by the environment.
    """

    subsidy: float
    cost: float
    action_component: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.subsidy <= MAX_SUBSIDY:
            raise ValueError(
                f"subsidy must be in [0, {MAX_SUBSIDY}], got {self.subsidy}"
            )
        if not 0.0 <= self.cost <= 1.0:
            raise ValueError(f"cost must be in [0, 1], got {self.cost}")

    # --- optimizer-space API -------------------------------------------------

    @property
    def dimension(self) -> int:
        """Always ``1``: only the subsidy rate is optimized."""
        return 1

    def encode(self) -> np.ndarray:
        """Return ``[subsidy / MAX_SUBSIDY]`` as a ``float32`` vector in ``[0, 1]``."""
        return np.array([self.subsidy / MAX_SUBSIDY], dtype=np.float32)

    def decode(self, x: np.ndarray) -> Self:
        """Return a copy with ``subsidy = x[0] * MAX_SUBSIDY`` (``cost`` is kept)."""
        x = self._validate(x)
        return replace(self, subsidy=float(x[0]) * MAX_SUBSIDY)

    def clip(self) -> Self:
        """Return a copy with ``subsidy`` clipped to ``[0, MAX_SUBSIDY]``."""
        return replace(self, subsidy=float(np.clip(self.subsidy, 0.0, MAX_SUBSIDY)))

    def param_names(self) -> list[str]:
        """Return ``["restoration_subsidy"]``."""
        return ["restoration_subsidy"]

    def to_vector(self) -> np.ndarray:
        """Expose the encoded (normalized) subsidy rate to agents.

        This is :meth:`encode`, i.e. ``subsidy / MAX_SUBSIDY``, not the raw
        rate; ``cost`` is not exposed.
        """
        return self.encode()

    # --- channels -------------------------------------------------------------

    @override(Mechanism)
    def reward(self, reward_dict: MultiAgentDict, **kwargs: Any) -> MultiAgentDict:
        """Add ``subsidy * e_i - cost * e_i**2`` to each agent's reward.

        Consumes ``action_after`` from ``kwargs``: the delivered actions
        supplied by the environment (not a binding), from which the effort
        ``e_i`` is read at ``action_component``. Returns Python floats.
        """
        actions = kwargs["action_after"]
        shaped: MultiAgentDict = {}
        for agent_id, reward in reward_dict.items():
            effort = float(
                np.asarray(actions[agent_id]).reshape(-1)[self.action_component]
            )
            shaped[agent_id] = float(
                reward + self.subsidy * effort - self.cost * effort**2
            )
        return shaped
