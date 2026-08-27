"""Social observation augmentation.

Each agent's observation is extended with the previous-step actions of all
other agents, in the order of ``agent_ids`` with the agent itself excluded:

    o*_i = [o_i, a_{1,t-1}, ..., a_{j,t-1}, ...]   for j != i

For ``N`` agents and ``d``-dimensional actions this adds ``(N - 1) * d``
features per agent.

This is observation shaping only. The counterfactual social-influence reward
of Jaques et al. (2019, PMLR 97:3040-3049), ``r_i + beta * sum_j D_KL[...]``,
is **not** implemented; ``influence_weight`` is reserved for it and currently
has no effect. The mechanism is fixed (``dimension == 0``).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Self

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class SocialInfluenceMechanism(Mechanism):
    """Expose peers' previous actions in each agent's observation.

    Parameters
    ----------
    influence_weight : float
        Reserved for the Jaques et al. KL reward bonus; unused for now.
    bindings : dict
        Must provide ``"previous_actions"``: ``env -> MultiAgentDict`` of the
        last delivered actions, and ``"agent_ids"``: ``env -> sequence`` fixing
        the peer ordering.
    """

    influence_weight: float = 0.0

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        missing = {"previous_actions", "agent_ids"} - set(self.bindings)
        if missing:
            raise ValueError(f"Missing social-influence bindings: {sorted(missing)}")
        if self.influence_weight < 0.0:
            raise ValueError("influence_weight must be non-negative.")

    # --- optimizer-space API -------------------------------------------------

    @property
    def dimension(self) -> int:
        return 0

    def encode(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    def decode(self, x: np.ndarray) -> Self:
        self._validate(x)
        return self

    def clip(self) -> Self:
        return self

    def param_names(self) -> list[str]:
        return []

    def to_vector(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    # --- channels -------------------------------------------------------------

    @override(Mechanism)
    def observation(self, observation_dict: MultiAgentDict, **kwargs) -> MultiAgentDict:
        previous_actions = kwargs["previous_actions"]
        agent_ids = list(kwargs["agent_ids"])

        augmented: MultiAgentDict = {}
        for agent_id, observation in observation_dict.items():
            peer_actions = [
                np.asarray(previous_actions[other_id], dtype=np.float32).reshape(-1)
                for other_id in agent_ids
                if other_id != agent_id
            ]
            augmented[agent_id] = np.concatenate(
                [np.asarray(observation, dtype=np.float32).reshape(-1), *peer_actions]
            ).astype(np.float32, copy=False)
        return augmented
