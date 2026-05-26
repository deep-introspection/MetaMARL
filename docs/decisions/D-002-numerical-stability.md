# D-002 — Numerical stability: raise on negative biomass instead of clamping

> **Date**: 2026-05-25
> **Status**: Accepted (recorded retroactively for Brick 1)
> **Brick**: 1 — Pure ecological model
> **Decider**: Rémy Ramadour (with Claude proposal)

## Context

ODE integrators (explicit Euler, RK45) can produce **non-physical negative
biomass** values for fish or algae under:

- A large `dt` combined with the explicit Euler method (numerical
  instability).
- Extreme harvest rates that exceed the endogenous production of the
  system.

The master codebase responded to this with `np.clip(fish_next, 0, max_fish)`
after each step. That clamp **silently** hid integrator failures and
produced misleading trajectories (a flat zero-floor that looks like a
stable state but is actually numerical garbage).

## Decision

The new `step()` function in `src/bilevel_fishery/ecology/dynamics.py`
**raises `EcologyInstabilityError`** whenever either the fish or algae
value comes out negative.

```python
if fish_next < 0 or algae_next < 0:
    raise EcologyInstabilityError(
        f"Negative biomass after step: fish={fish_next:.6g}, "
        f"algae={algae_next:.6g}. Reduce dt or use integrator='rk45'."
    )
```

## Alternatives considered

| Option | Behaviour | Trade-off |
|---|---|---|
| A — Silent clamp (master) | `max(0, x)` | Hides bugs, trajectory looks plausible |
| B — Soft clamp to `eps` + log warning | `max(eps, x) + log.warn(...)` | Warnings get ignored in long runs |
| C — Switch to a positivity-preserving solver | Implicit / operator splitting | Large refactor for marginal gain |
| **D — Raise (selected)** | `raise EcologyInstabilityError(...)` | Fails loud, forces caller awareness |

## Rationale

- **Scientific rigour**: negative biomass is non-physical. Continuing the
  simulation past that point is just lying with numbers.
- **Separation of concerns**: the caller has more context than the dynamics
  module. The caller can decide to truncate, switch solver, or fail the
  experiment. The dynamics module only knows about the equations.
- **Physical cap belongs upstream**: `FisheryEnv` (Brick 2) applies a
  *physical* cap (`harvest_realized = min(demanded, 0.99 * fish / dt)`)
  **before** calling `step()`, so it never triggers this error in normal
  use. That cap is semantically different from a numerical clamp: it
  encodes "you cannot fish what doesn't exist", not "we'll pretend bad
  numbers don't exist".

## Consequences

- Callers that pass aggressive harvest values to `step()` directly must
  wrap in `try/except`. Notebooks `01_ecology.ipynb` and
  `02_environment.ipynb` do this for the pedagogical over-harvest and
  Euler-divergence demos.
- If we add a new solver (Schaefer logistic, implicit Euler, etc.), it
  must honour the same `raise on negative` contract.
- The Brick 1 notebook teaches this explicitly via the `over-harvest` cell:
  the user *sees* the system collapse and the `EcologyInstabilityError`
  raised at a precise time step.

## Implementation

`src/bilevel_fishery/ecology/dynamics.py` — the lines after `_step_rk45` /
`_step_euler` that check `fish_next < 0 or algae_next < 0`.

## References

- Reverse-prompt audit table for Brick 1, row B1.5: see
  [docs/bricks/01_ecology.md](../bricks/01_ecology.md).
- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary
  Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
  (RK45 positivity properties for well-posed parameters.)
