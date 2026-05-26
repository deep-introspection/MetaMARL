"""Single-agent Gymnasium environment wrapping the pure ecology model.

The environment is intentionally minimal: one agent, one continuous action
(fishing intensity), no regulation mechanism, no World shared state, no Ray.
Those concerns layer on top in subsequent bricks.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from bilevel_fishery.ecology import (
    EcologicalState,
    EcologyInstabilityError,
    EcologyParams,
    reset_state,
    step,
)

Observation = NDArray[np.float32]
Action = NDArray[np.float32]


class FisheryEnv(gym.Env[Observation, Action]):
    r"""Single-agent fishery environment without a regulation mechanism.

    Parameters
    ----------
    params
        Ecological parameters. ``None`` uses :class:`EcologyParams` defaults.
    max_harvest_rate
        Maximum harvest rate the agent can demand (biomass per unit time)
        when ``action = 1``. The realized harvest is capped to keep the
        post-step biomass strictly positive (see Notes).
    horizon
        Number of steps before ``truncated=True``.

    Notes
    -----
    **Spaces**

    - Observation: ``Box(0, 1, shape=(2,))`` — ``(fish/max_fish, algae/max_algae)``.
    - Action: ``Box(0, 1, shape=(1,))`` — fishing intensity, normalized.

    **Reward**

    .. math::

        r_t = \\log(1 + H_t^{\\mathrm{realized}})

    Concave (decreasing marginal utility): encourages steady catches over
    bursty over-harvesting. See ``docs/decisions/D-001-reward-function.md``.

    **Physical harvest cap**

    The demanded harvest is ``action[0] * max_harvest_rate``; the realized
    one is

    .. math::

        H^{\\mathrm{realized}}
        = \\min(H^{\\mathrm{demanded}},\\ 0.99 \\cdot F / \\Delta t)

    The cap models the physical impossibility of harvesting more biomass
    than what exists. It is NOT the silent ``np.clip`` of the master
    codebase — that one was hiding numerical instability of Euler. Here
    the cap is a deliberate sematic statement of the environment.

    **No termination**

    ``terminated`` is always ``False``. The episode only ends via
    ``truncated`` when the horizon is reached. Stock collapse is part of
    the dynamics, not an environment-level termination signal.
    """

    # gymnasium.Env declares ``metadata`` as a plain class attribute (not
    # ClassVar), so we mirror that to keep mypy happy. RUF012 is a false
    # positive here: we never mutate this dict.
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        params: EcologyParams | None = None,
        max_harvest_rate: float = 2.0,
        horizon: int = 200,
    ) -> None:
        super().__init__()
        self.params = params if params is not None else EcologyParams()
        self.max_harvest_rate = float(max_harvest_rate)
        self.horizon = int(horizon)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._state: EcologicalState | None = None
        self._t: int = 0
        self._rng: np.random.Generator | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        seed
            Seed used both by the gymnasium base class and by
            :func:`reset_state` for log-normal initial-state noise.
        options
            Unused; reserved for the gymnasium API.
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._state = reset_state(self.params, self._rng)
        self._t = 0
        return self._observation(), {}

    def step(
        self,
        action: Action,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the Gymnasium 5-tuple.

        Returns ``(obs, reward, terminated, truncated, info)``.
        """
        if self._state is None:
            raise RuntimeError("FisheryEnv: call reset() before step().")

        harvest_demanded = float(action[0]) * self.max_harvest_rate
        max_extractable = 0.99 * self._state.fish / self.params.dt
        harvest_realized = float(min(harvest_demanded, max(0.0, max_extractable)))

        try:
            self._state = step(self._state, self.params, harvest=harvest_realized)
        except EcologyInstabilityError:
            # The harvest cap is a heuristic and can be a hair too loose near
            # total stock depletion (RK45 returns a tiny negative fish biomass
            # due to numerical noise). The env-level semantics is "you cannot
            # fish what doesn't exist": treat as full collapse with no harvest.
            self._state = EcologicalState(fish=0.0, algae=self._state.algae)
            harvest_realized = 0.0
        self._t += 1

        reward = float(np.log1p(harvest_realized))
        terminated = False
        truncated = self._t >= self.horizon
        info: dict[str, Any] = {
            "harvest_demanded": harvest_demanded,
            "harvest_realized": harvest_realized,
            "fish": self._state.fish,
            "algae": self._state.algae,
        }
        return self._observation(), reward, terminated, truncated, info

    def _observation(self) -> Observation:
        assert self._state is not None
        fish_norm = self._state.fish / self.params.max_fish
        algae_norm = self._state.algae / self.params.max_algae
        return np.array([fish_norm, algae_norm], dtype=np.float32)
