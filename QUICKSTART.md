# Quickstart — from a clone to a first run in ten minutes

This walkthrough runs the fishery benchmark end-to-end with the explicit
mechanism abstraction of this branch (quota, restoration subsidy and social
observation), on a laptop, without a Weights & Biases account.

## 1. Install

```bash
git clone https://github.com/deep-introspection/bilevel-fishery.git
cd bilevel-fishery
git checkout feature/social-influence/testing
uv sync --group dev
```

## 2. Check the toolchain

```bash
uv run python -m pytest -m "not integration and not notebook"
```

About 150 unit tests run in a few seconds, with a coverage table for `core/`.

## 3. Run a short experiment

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 \
    --num-candidates 2 --num-eval-seeds 1
```

This starts Ray locally, builds two candidate mechanisms (a quota level and a
subsidy rate each), trains two fishers for two APPO iterations per candidate,
evaluates them, and runs two ES generations. It takes about a minute. The log
shows the per-parameter ES gradients and ends with:

```
[ES] gen=2 | best=... | mean=...+/-... | sigma=0.1500
[Bilevel] Run finished | iters=2 | converged=False | best_fitness=...
```

Plots go to an offline W&B run under `wandb/` (`wandb sync` uploads it later).

## 4. What is being optimized

`examples/bilevel_fishery/debug.py::build_mechanism` composes the regulation:

```python
ChainedMechanism(children=(
    QuotaMechanism(fixed_quota=0.56, bindings={"resource_level": lambda env: env.S_t["fish"] / env.K}),
    SubsidyMechanism(subsidy=0.10, cost=0.05, action_component=1),
    SocialInfluenceMechanism(bindings={"previous_actions": ..., "agent_ids": ...}),
))
```

The quota caps the harvest fraction (action component 0) as the stock falls,
the subsidy rewards restoration effort (action component 1), and the social
mechanism shows each fisher the other fishers' last actions. `fixed_quota` and
`subsidy` are the two optimized parameters (`mechanism.param_names()`); the
other two mechanisms are fixed. `BilevelConfig().mechanism(mechanism=...)`
hands the composite to both levels.

## 5. Learn the mechanics

- `tutorials/mechanism_algorithms.ipynb` — each mechanism by hand, composition,
  a real environment rollout against a `World` actor (about ten seconds).
- `tutorials/custom_benchmark_creation.ipynb` — writing a new benchmark with
  the hook decorators and a custom mechanism, testing the equations before
  any RL.
- `docs/ARCHITECTURE.md` — the run assembly and the invariants.

## 6. Scale up

Drop the flags to get the reference configuration (10 fishers, 4 candidates,
50 APPO iterations per candidate, 1000 generations, 3 evaluation seeds); use
`--no-social` to drop the social observation and `--project` for a real W&B
project. `TODO.md` lists what remains (quota parity against `dev`, per-step
mechanism context).
