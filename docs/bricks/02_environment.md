# Brick 2 — Single-agent Gymnasium environment

> **Date**: 2026-05-25
> **Branch**: `rebuild/from-scratch`
> **Reference**: master audit `pre-rebuild-2026-05-25` —
> `core/envs/base.py`, `core/envs/regulated.py`,
> `examples/cartpole/regulated_env.py`.

## Why this brick

We wrap `ecology.step()` (Brick 1) in a **standard Gymnasium** single-agent
interface: `reset()` / `step()` / `observation_space` / `action_space`.

The result is a simple fisher who can fish in a lake, with:

- No dependency on Ray, World, mechanism, or multi-agent
- A Gymnasium 1.x compliant API
- A physical cap on harvest (you cannot fish what doesn't exist)
- A concave reward (`log1p`) — decision tracked in D-001

This is the building block on which the next bricks graft:

- Brick 3: a regulation mechanism (quota, fine)
- Brick 4: several fishers (multi-agent)
- Brick 5: a shared `World` for bilevel optimizers

## Reverse-prompts and audit corrections

| # | Source prompt | Correction |
|---|---|---|
| B2.1 | `BaseEnv` coupled to a `World` Ray actor | **Removed** — env is autonomous |
| B2.2 | `RegulatedEnv` fetches mechanism via Ray | **Removed** — no mechanism in Brick 2 |
| B2.3 | `MultiAgentRegulatedEnv` via RLlib | **Removed** — pure single-agent |
| B2.4 | `CartpoleRegulatedEnv` "single-agent" via multi-agent inheritance | **Reproduced** as direct `gymnasium.Env`, no exotic inheritance |

## Design decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Action space | `Box(0, 1, shape=(1,), float32)` — normalized intensity |
| 2 | Observation space | `Box(0, 1, shape=(2,), float32)` — `(fish/max_fish, algae/max_algae)` |
| 3 | Reward function | `log(1 + harvest_realized)` — concave (see [D-001](../decisions/D-001-reward-function.md)) |
| 4 | Physical cap | `harvest_realized = min(action·max_rate, 0.99·fish/dt)` |
| 5 | Horizon | 200 steps default, configurable |
| 6 | Terminated | Always `False` (stock can collapse without ending the episode) |
| 7 | Truncated | At `horizon` |
| 8 | Seeding | Standard `gymnasium` + Brick 1 `reset_state` |

## Decision log — NOTE FOR NADINE

[D-001 Reward function](../decisions/D-001-reward-function.md) — the choice
of `log1p(harvest)` as the single-agent fisher reward. If we change the
reward later, previously trained agents will **no longer be comparable**.

## Concepts introduced

- **Gymnasium 1.x API**: `reset(seed) -> (obs, info)`, `step(action) ->
  (obs, reward, terminated, truncated, info)` (5-tuple, not the legacy
  4-tuple of `gym`)
- **`Box` spaces**, continuous, normalized to `[0, 1]`
- **Physical cap** vs **numerical clamp** (different semantics)
- **Concave reward** (CRRA with η=1, logarithmic utility)
- **Self-contained environment** without external shared state

## Layout added

```
src/bilevel_fishery/envs/
├── __init__.py
└── fishery_env.py          FisheryEnv (gymnasium.Env)

tests/envs/
├── __init__.py
└── test_fishery_env.py     11 API + behaviour tests

config/env_default.yaml     Default parameters
docs/decisions/D-001-...    ADR on reward choice
notebooks/02_environment.ipynb  Deep dive (~12 figures): reward function
                                shape, 30-seed stochastic bands, action
                                sweep (trajectories, finals, Pareto,
                                heatmap), physical cap recovery,
                                max_harvest_rate sensitivity
```

## Verifications

```bash
make test            # 31 tests = 20 (Brick 1) + 11 (Brick 2), ~98% coverage
make lint            # ruff strict clean
make typecheck       # mypy strict clean
make notebook-test   # 00 + 01 + 02 execute end-to-end
```

## What is NOT part of Brick 2

- No regulation mechanism (quota, fine, ban) — Brick 3
- No multi-agent — Brick 4
- No `World`, no Ray — Brick 5
- No RL training (PPO) — Brick 6

## Next brick

**Brick 3** — Mechanism design: introduce regulation (fixed quota,
stock-proportional quota, fine, minimum threshold). `FisheryEnv` is extended
to integrate those constraints in the reward computation
(net utility = harvest − fine·violation).
