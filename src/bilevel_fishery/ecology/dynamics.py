"""Lotka-Volterra dynamics for the bilevel-fishery ecological model.

This module exposes a pure :func:`step` function that advances the
predator-prey system by ``params.dt`` units of time. Two integrators are
available:

- ``rk45`` (default): adaptive Runge-Kutta 4(5) via
  :func:`scipy.integrate.solve_ivp`. Robust, accurate, guarantees positivity
  for well-posed parameters (Hairer et al., 1993).
- ``euler``: explicit Euler. Simple and pedagogical, but unstable for large
  ``dt`` and prone to producing negative biomass — in which case
  :class:`EcologyInstabilityError` is raised so the caller cannot silently
  drift into a non-physical regime.

The dynamics are deterministic; the only randomness lives in initial
conditions (see :func:`reset_state`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

from bilevel_fishery.ecology.params import EcologyParams
from bilevel_fishery.ecology.state import EcologicalState

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EcologyInstabilityError(RuntimeError):
    """Raised when the ODE integrator returns negative biomass.

    This typically indicates the integrator is misconfigured (``dt`` too
    large with Euler) or the harvest pressure is non-physically high.
    """


def reset_state(params: EcologyParams, rng: np.random.Generator) -> EcologicalState:
    """Sample an initial ecological state, optionally with log-normal noise.

    Parameters
    ----------
    params
        Ecological parameters. ``params.noise_std`` controls noise magnitude:
        ``0.0`` returns the deterministic ``(fish_init, algae_init)`` state.
    rng
        Seeded NumPy random generator (for reproducibility).

    Returns
    -------
    EcologicalState
        Initial state with strictly positive biomass.
    """
    if params.noise_std == 0.0:
        return EcologicalState(fish=params.fish_init, algae=params.algae_init)

    eps = 1e-8
    fish = max(eps, rng.lognormal(np.log(params.fish_init), params.noise_std))
    algae = max(eps, rng.lognormal(np.log(params.algae_init), params.noise_std))
    return EcologicalState(fish=fish, algae=algae)


def step(
    state: EcologicalState,
    params: EcologyParams,
    harvest: float = 0.0,
) -> EcologicalState:
    r"""Advance the ecological state by ``params.dt`` units of time.

    Parameters
    ----------
    state
        Current ecological state (fish and algae biomass).
    params
        Lotka-Volterra parameters and integrator choice.
    harvest
        External fish removal rate (biomass per unit time). Default ``0.0``.

    Returns
    -------
    EcologicalState
        New state at :math:`t + dt`.

    Raises
    ------
    EcologyInstabilityError
        If the integrator returns a negative biomass.

    Notes
    -----
    Equations integrated:

    .. math::

        \frac{dF}{dt} = \delta\, A\, F - \gamma\, F - H,

        \frac{dA}{dt} = \alpha\, A - \beta\, A\, F.
    """
    if params.integrator == "rk45":
        fish_next, algae_next = _step_rk45(state, params, harvest)
    else:  # "euler" — Literal guards against other values
        fish_next, algae_next = _step_euler(state, params, harvest)

    if fish_next < 0 or algae_next < 0:
        raise EcologyInstabilityError(
            f"Negative biomass after step: fish={fish_next:.6g}, "
            f"algae={algae_next:.6g}. Reduce dt or use integrator='rk45'."
        )

    return EcologicalState(fish=fish_next, algae=algae_next)


def _lotka_volterra_rhs(
    t: float,
    y: NDArray[np.float64],
    alpha: float,
    beta: float,
    delta: float,
    gamma: float,
    harvest: float,
) -> NDArray[np.float64]:
    """Right-hand side of the Lotka-Volterra ODE (signature for ``solve_ivp``)."""
    del t  # autonomous system
    fish, algae = y
    dfish = delta * algae * fish - gamma * fish - harvest
    dalgae = alpha * algae - beta * algae * fish
    return np.array([dfish, dalgae])


def _step_rk45(
    state: EcologicalState, params: EcologyParams, harvest: float
) -> tuple[float, float]:
    """Single integration step using adaptive RK45 (SciPy)."""
    sol = solve_ivp(
        _lotka_volterra_rhs,
        t_span=(0.0, params.dt),
        y0=np.array([state.fish, state.algae]),
        method="RK45",
        args=(params.alpha, params.beta, params.delta, params.gamma, harvest),
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise EcologyInstabilityError(f"RK45 solver failed: {sol.message}")
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def _step_euler(
    state: EcologicalState, params: EcologyParams, harvest: float
) -> tuple[float, float]:
    """Single integration step using explicit Euler (pedagogical)."""
    fish, algae = state.fish, state.algae
    dfish = params.delta * algae * fish - params.gamma * fish - harvest
    dalgae = params.alpha * algae - params.beta * algae * fish
    fish_next = fish + params.dt * dfish
    algae_next = algae + params.dt * dalgae
    return fish_next, algae_next
