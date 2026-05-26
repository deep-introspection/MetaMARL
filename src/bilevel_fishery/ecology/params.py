"""Parameters of the Lotka-Volterra ecological model.

The default values place the deterministic equilibrium at
``(F*, A*) = (alpha/beta, gamma/delta) = (10, 20)``, which keeps trajectories
in a regime where both populations remain numerically well-behaved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EcologyParams(BaseModel):
    r"""Parameters of the predator-prey Lotka-Volterra fishery model.

    The dynamical system is

    .. math::

        \frac{dF}{dt} = \delta\, A\, F - \gamma\, F - H,

        \frac{dA}{dt} = \alpha\, A - \beta\, A\, F,

    where :math:`F` is fish biomass (predator), :math:`A` is algae biomass
    (prey), :math:`H` is external harvest, and
    :math:`\alpha, \beta, \delta, \gamma > 0`.

    Parameters
    ----------
    alpha
        Intrinsic algae growth rate (per unit time).
    beta
        Algae predation rate by fish (per fish per unit time).
    delta
        Fish growth efficiency per algae consumed (per algae per unit time).
    gamma
        Natural fish mortality rate (per unit time).
    fish_init
        Initial fish biomass at reset (before noise).
    algae_init
        Initial algae biomass at reset (before noise).
    max_fish
        Reference scale for fish (used for normalization in later bricks).
    max_algae
        Reference scale for algae.
    dt
        Integration time step (used by the Euler integrator; RK45 adapts
        internally and exits after ``dt``).
    noise_std
        Log-normal standard deviation applied to the initial state at reset.
        ``0.0`` disables the noise (purely deterministic initial conditions).
    integrator
        ODE solver. ``"rk45"`` (default) is adaptive RK4(5) from SciPy;
        ``"euler"`` is explicit Euler, retained for pedagogical comparison.

    References
    ----------
    Lotka, A. J. (1925). *Elements of Physical Biology.* Williams & Wilkins.

    Volterra, V. (1926). Fluctuations in the abundance of a species
    considered mathematically. *Nature*, 118, 558-560.

    Clark, C. W. (1990). *Mathematical Bioeconomics: The Optimal Management
    of Renewable Resources* (2nd ed.). Wiley.
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(default=1.0, gt=0)
    beta: float = Field(default=0.1, gt=0)
    delta: float = Field(default=0.075, gt=0)
    gamma: float = Field(default=1.5, gt=0)

    fish_init: float = Field(default=10.0, gt=0)
    algae_init: float = Field(default=10.0, gt=0)

    max_fish: float = Field(default=100.0, gt=0)
    max_algae: float = Field(default=100.0, gt=0)

    dt: float = Field(default=0.01, gt=0)
    noise_std: float = Field(default=0.05, ge=0)
    integrator: Literal["rk45", "euler"] = "rk45"

    @model_validator(mode="after")
    def _initial_under_max(self) -> EcologyParams:
        if self.fish_init > self.max_fish:
            raise ValueError(
                f"fish_init={self.fish_init} exceeds max_fish={self.max_fish}"
            )
        if self.algae_init > self.max_algae:
            raise ValueError(
                f"algae_init={self.algae_init} exceeds max_algae={self.max_algae}"
            )
        return self
