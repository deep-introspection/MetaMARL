# D-001 — Reward function of `FisheryEnv` (single-agent)

> **Date**: 2026-05-25
> **Status**: Accepted
> **Brick**: 2 — Single-agent Gymnasium environment
> **Decider**: Rémy Ramadour (with Claude proposal)
> **Note for Nadine**: this choice conditions every reward value a fisher
> trained in this brick will see. If we change the reward later,
> previously trained agents are **no longer comparable**.

## Context

The fisher in `FisheryEnv` must receive a **reward** at each step that
represents the economic value of its action. The reward function shapes
the behaviour learned by the RL agent.

## Decision

```python
reward = log(1 + harvest_realized)
```

A **concave** reward (logarithmic utility) on the realized catch.

## Alternatives considered

| Option | Formula | Pros | Cons |
|---|---|---|---|
| A — Linear | `harvest_realized` | Simple, interpretable | No risk aversion, encourages bursts |
| B — Utility scaling (master) | `harvest_realized * fish/max_fish` | Penalizes scarcity | Coupled to stock, less readable |
| **C — Concave log (selected)** | `log(1 + harvest_realized)` | Risk aversion, regular catches, economically standard | Slightly more complex |

## Rationale

Concavity (diminishing marginal returns):

- Encourages the agent to **spread** catches rather than make sporadic
  large hauls.
- Is consistent with standard utility theory (risk aversion).
- Provides a **stable** learning signal: does not blow up for large
  harvests, does not vanish for small ones.

## Consequences

- The RL agent will learn a **smoothing** strategy (preference for
  regularity).
- Reward values lie in `[0, log(1 + max_harvest_rate * dt)]` ≈ `[0, 0.7]`
  for default parameters.
- Comparing two agents trained with different reward functions is **not
  meaningful** without normalization.

## Implementation

`src/bilevel_fishery/envs/fishery_env.py:step()`, the line that calls
`np.log1p`.

## Academic references

- Pratt, J. W. (1964). Risk aversion in the small and in the large.
  *Econometrica*, 32(1/2), 122-136.
- Arrow, K. J. (1965). *Aspects of the Theory of Risk Bearing*.
  Yrjö Jahnssonin Säätiö, Helsinki.

(Logarithmic utility ≡ constant relative risk aversion, CRRA, η=1.)
