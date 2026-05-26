# Brick 1 — Pure ecological model

> **Date**: 2026-05-25
> **Branch**: `rebuild/from-scratch`
> **Reference**: master audit `pre-rebuild-2026-05-25`,
> file `examples/bilevel_fishery/regulated_env.py` (`transition_kernel` method).

## Why this brick

We extract the **pure physics** of the fishery model: the fish + algae
dynamics, with no coupling to Gym, Ray, agents, or regulation mechanism.

Benefits:

- **Independently testable**: 18 unit tests cover equilibria, positivity,
  conservation, and integrator parity.
- **Reusable**: we can plug a Gym wrapper on top (Brick 2), an alternative
  solver (Schaefer), or a Numba/JAX benchmark, without touching the physics.
- **Readable**: the dynamics fit in 4 lines (vs 366 lines in master).

## Reverse-prompts and audit corrections

| # | Source prompt | Correction |
|---|---|---|
| B1.1 | Predator-prey Lotka-Volterra | **Cite** Lotka (1925), Volterra (1926), Clark (1990) |
| B1.2 | Params in `ecology_cfg: dict` | **Pydantic** `EcologyParams`, frozen + validated |
| B1.3 | Log-normal noise at reset | **`noise_std` parameter** (vs magic number 0.05) |
| B1.4 | Explicit Euler | **RK45 by default** + Euler kept as pedagogical option |
| B1.5 | Clamp to `[0, max]` | **No clamp** + `EcologyInstabilityError` if drift |
| B1.6 | Return `dict[str, float]` | **`EcologicalState`** frozen+slots dataclass |
| B1.7 (added) | Separate physics from Gym | Standalone `ecology/dynamics.py` module |
| B1.8 (added) | Numerical property tests | 18 tests in `tests/ecology/` |
| B1.9 (added) | Cite the references | `ecology/references.md` + docstrings |

## Design decisions (validated with Rémy)

| Decision | Choice |
|---|---|
| ODE solver | RK45 default + Euler option |
| Harvest API | `harvest: float` passed to `step()` |
| State representation | `@dataclass(frozen=True, slots=True)` |
| Stability | RK45 keeps positivity; otherwise `EcologyInstabilityError` |

## Concepts introduced

- **Predator-prey ODE system**: coupled F (fish) and A (algae)
- **Numerical integration**: explicit Euler vs adaptive RK45
- **Non-trivial equilibrium**: `(F*, A*) = (α/β, γ/δ) = (10, 20)` at defaults
- **Pure function**: `step(state, params, harvest) → state`, no `self`
- **Immutability**: `@dataclass(frozen=True, slots=True)` + Pydantic `frozen=True`
- **Pydantic validation**: `gt=0` constraints, cross-field `model_validator`
- **Property tests**: stationary equilibrium, exponential decay without
  algae, exponential growth without fish

## Verifications

```bash
make test            # 20 tests pass, ~96% coverage on ecology/
make lint            # ruff clean
make typecheck       # mypy strict clean
make notebook-test   # 00 + 01 execute end-to-end
```

## Final layout

```
src/bilevel_fishery/ecology/
├── __init__.py       # re-export public API
├── params.py         # EcologyParams (Pydantic, frozen)
├── state.py          # EcologicalState (frozen+slots dataclass)
├── dynamics.py       # step() + integrators + EcologyInstabilityError
└── references.md     # citations + justification

tests/ecology/
├── test_params.py        # 7 tests
├── test_dynamics.py      # 9 tests
└── test_solver_parity.py # 2 tests

config/ecology_default.yaml  # documented default parameters
notebooks/01_ecology.ipynb   # interactive exploration
```

## What is NOT part of Brick 1

- No Gym API (comes in Brick 2: single-agent environment)
- No fishers (Brick 2)
- No regulation mechanism (Brick 3)
- No linear stability / Jacobian analysis (possibly Brick 1.5)

## Next brick

**Brick 2** — Single-agent Gymnasium environment: wrap the ecological
dynamics in the standard `reset() / step() / observation_space /
action_space` API. The first real fisher (single), with scripted behaviour
for now.
