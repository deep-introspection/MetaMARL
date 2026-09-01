"""Smooth harvest quota acting on one action component.

The quota converts the current normalized resource level ``b`` into an allowed
fraction of the maximal request and softly caps every agent's requested
fraction at that value. With ``sigma`` the logistic function, ``q`` the quota
parameter and ``w_q`` the transition width:

    L = sigma((0 - q) / w_q),  U = sigma((1 - q) / w_q),  C = sigma((b - q) / w_q)
    allowed_frac = (C - L) / (U - L)

so that ``allowed_frac`` is 0 when the resource is depleted, 1 when it is at
carrying capacity, and transitions smoothly around ``b = q``. The requested
fraction ``u`` on ``action_component`` becomes

    u* = u - smooth_plus(u - allowed_frac; w_u)

which leaves requests below the allowance unchanged and smoothly caps the rest.
The only optimized parameter is ``fixed_quota``.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Self

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict
from core.utils import sigmoid, smooth_positive_zero_at_origin

EPS = 1e-8


@dataclass(frozen=True)
class QuotaMechanism(Mechanism):
    """Smooth quota on the harvest fraction (see module docstring).

    Parameters
    ----------
    fixed_quota : float
        Resource level, in ``[0, 1]``, around which the allowance transitions.
        Optimized parameter (``dimension == 1``).
    bindings : dict
        Must provide ``"resource_level"``: ``env -> float`` in ``[0, 1]``.
    action_component : int
        Index of the action component holding the requested fraction.
    quota_transition_width, usage_transition_width : float
        Widths of the two logistic transitions (resource axis, request axis).

    Notes
    -----
    ``allowed_frac`` computed in :meth:`action` is cached on the instance so
    that :meth:`observation` can expose it to agents in the same step. A
    mechanism instance therefore carries per-step state and must not be shared
    across concurrently stepping environments.
    """

    fixed_quota: float

    bindings: dict[str, Callable[[Any], Any]] = field(repr=False, compare=False)
    action_component: int = 0

    # Fixed algorithmic parameters.
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005

    _context: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        missing = {"resource_level"} - set(self.bindings)
        if missing:
            raise ValueError(f"Missing quota bindings: {sorted(missing)}")
        if not 0.0 <= self.fixed_quota <= 1.0:
            raise ValueError(f"fixed_quota must be in [0, 1], got {self.fixed_quota}")
        if self.quota_transition_width <= 0.0:
            raise ValueError("quota_transition_width must be positive")
        if self.usage_transition_width <= 0.0:
            raise ValueError("usage_transition_width must be positive")

    # --- optimizer-space API -------------------------------------------------

    @property
    def dimension(self) -> int:
        """Always ``1``: only ``fixed_quota`` is optimized."""
        return 1

    def encode(self) -> np.ndarray:
        """Return ``[fixed_quota]`` as a ``float32`` vector (already in ``[0, 1]``)."""
        return np.array([self.fixed_quota], dtype=np.float32)

    def decode(self, x: np.ndarray) -> Self:
        """Return a copy with ``fixed_quota`` set to ``x[0]`` (bindings are kept)."""
        x = self._validate(x)
        return replace(self, fixed_quota=float(x[0]))

    def clip(self) -> Self:
        """Return a copy with ``fixed_quota`` clipped to ``[0, 1]``."""
        return replace(self, fixed_quota=float(np.clip(self.fixed_quota, 0.0, 1.0)))

    def param_names(self) -> list[str]:
        """Return ``["fixed_quota"]``."""
        return ["fixed_quota"]

    def to_vector(self) -> np.ndarray:
        """Expose ``[fixed_quota]`` to agents; identical to :meth:`encode`."""
        return np.array([self.fixed_quota], dtype=np.float32)

    def observation_names(self) -> list[str]:
        """Name of the feature :meth:`observation` appends: ``["effective_quota"]``."""
        return ["effective_quota"]

    # --- channels -------------------------------------------------------------

    def allowed_fraction(self, resource_level: float) -> float:
        """Allowed request fraction for a normalized resource level."""
        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - self.fixed_quota) / width)
        upper = sigmoid((1.0 - self.fixed_quota) / width)
        current = sigmoid((float(resource_level) - self.fixed_quota) / width)
        return float((current - lower) / max(upper - lower, EPS))

    @override(Mechanism)
    def action(self, action_dict: MultiAgentDict, **kwargs: Any) -> MultiAgentDict:
        """Softly cap every agent's requested fraction at the current allowance.

        Consumes the ``resource_level`` binding from ``kwargs`` to compute
        ``allowed_frac``, then rewrites ``action_component`` of each action as
        ``u - smooth_plus(u - allowed_frac)``; the other components are copied
        unchanged. The allowance and the requested/delivered action
        dictionaries are cached in ``_context`` for :meth:`observation` and for
        diagnostics.
        """
        allowed_frac = self.allowed_fraction(kwargs["resource_level"])
        self._context["allowed_frac"] = allowed_frac

        requested: MultiAgentDict = {}
        regulated: MultiAgentDict = {}

        for agent_id, action in action_dict.items():
            action = np.asarray(action, dtype=np.float32)
            requested[agent_id] = action.copy()
            regulated_action = action.copy()
            requested_frac = float(action[self.action_component])
            regulated_action[self.action_component] = (
                requested_frac
                - smooth_positive_zero_at_origin(
                    requested_frac - allowed_frac, self.usage_transition_width
                )
            )
            regulated[agent_id] = regulated_action

        self._context["requested_action_dict"] = requested
        self._context["delivered_action_dict"] = regulated
        return regulated

    @override(Mechanism)
    def observation(
        self, observation_dict: MultiAgentDict, **kwargs: Any
    ) -> MultiAgentDict:
        """Append the current ``allowed_frac`` to every agent observation.

        Uses the value cached by :meth:`action` in this step; before the first
        action of an episode (reset) it is recomputed from ``resource_level``
        so that the observation size is constant. Without either, the
        observation is returned unchanged.
        """
        allowed_frac = self._context.get("allowed_frac")
        if allowed_frac is None and "resource_level" in kwargs:
            allowed_frac = self.allowed_fraction(kwargs["resource_level"])
        if allowed_frac is None:
            return observation_dict
        extra = np.array([allowed_frac], dtype=np.float32)
        return {
            agent_id: np.concatenate(
                [np.asarray(obs, dtype=np.float32).reshape(-1), extra]
            )
            for agent_id, obs in observation_dict.items()
        }
