# Ecological model — scientific references

## The Lotka-Volterra predator-prey system

The ecological dynamics of the fishery are governed by the classical
Lotka-Volterra equations with an external harvest term:

$$
\begin{aligned}
\frac{dF}{dt} &= \delta A F - \gamma F - H \\
\frac{dA}{dt} &= \alpha A - \beta A F
\end{aligned}
$$

with $F$ the fish biomass (predator), $A$ the algae biomass (prey), $H$ the
external harvest rate, and $\alpha, \beta, \delta, \gamma > 0$.

Equilibria of the harvest-free system are at
$(F^\*, A^\*) = (\alpha/\beta, \gamma/\delta)$, around which trajectories form
closed orbits (neutrally stable centre).

**Foundational references**

- Lotka, A. J. (1925). *Elements of Physical Biology.* Williams & Wilkins.
- Volterra, V. (1926). Fluctuations in the abundance of a species considered
  mathematically. *Nature*, 118, 558-560.

## Why Lotka-Volterra and not Schaefer logistic?

The Schaefer (1954) monospecies model is the historical workhorse of
fisheries economics:

$$\frac{dS}{dt} = r\, S\, \left(1 - \frac{S}{K}\right) - H,$$

with intrinsic growth $r$ and carrying capacity $K$.

| Property | Schaefer | Lotka-Volterra |
|---|---|---|
| Species | 1 | 2 (extensible) |
| Equilibrium | Stable carrying capacity $K$ | Centre / closed orbits |
| Realism | Aggregate stock | Trophic web |
| Parameters | 2 ($r$, $K$) | 4 ($\alpha, \beta, \gamma, \delta$) |
| Pedagogy | Simpler dynamics | Richer dynamics |

This project uses Lotka-Volterra because:

1. The richer dynamics (oscillations, trophic coupling) make the regulator's
   mechanism-design problem more interesting.
2. The regulator's actions on fish biomass propagate to algae through the
   coupling, which is closer to the ecosystem-based fisheries management
   paradigm.

A future brick can add Schaefer as an alternative dynamics module behind the
same `step(state, params, harvest)` interface.

**Reference**

- Schaefer, M. B. (1954). Some aspects of the dynamics of populations
  important to the management of the commercial marine fisheries.
  *Bulletin of the Inter-American Tropical Tuna Commission*, 1, 25-56.
- Clark, C. W. (1990). *Mathematical Bioeconomics: The Optimal Management of
  Renewable Resources* (2nd ed.). Wiley.

## Numerical integration

Two integrators are available:

- **RK45** (default): adaptive Runge-Kutta 4(5) via
  `scipy.integrate.solve_ivp`. Robust to mild stiffness, controls local error
  via embedded estimates, and preserves positivity for well-posed parameters.
- **Euler explicit**: only retained for pedagogy, to demonstrate numerical
  instability when `dt` is too large or harvest pressure is high.

**Reference**

- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary
  Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
