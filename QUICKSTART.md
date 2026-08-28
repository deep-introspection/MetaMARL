# Quickstart — from a clone to a first run in ten minutes

This walkthrough runs the fishery benchmark end-to-end with the typed
metrics/reporting stack of this branch, on a laptop, without a Weights &
Biases account.

## 1. Install

```bash
git clone https://github.com/deep-introspection/bilevel-fishery.git
cd bilevel-fishery
git checkout feature/logging-testing
uv sync --group dev
```

## 2. Check the toolchain

```bash
uv run python -m pytest -m "not integration and not notebook"
```

About ninety unit tests run in a few seconds, with a coverage table for
`core/`.

## 3. Run a short experiment

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 \
    --num-candidates 2 --num-eval-seeds 1 --reporter csv --output-dir results
```

This starts Ray locally, builds two candidate quotas, trains two fishers for
two APPO iterations per candidate, evaluates them, and runs two ES
generations. It takes about a minute. The log ends with:

```
[ES] gen=2 | best=... | mean=...+/-... | sigma=0.1500
[Bilevel] Run finished | iters=2 | converged=False | best_fitness=...
```

## 4. Read the outputs

With `--reporter csv`, every query becomes one long-form CSV under
`results/bilevel/<world>-<owner>/<query title>.csv`, with columns
`query, x, series, value, error`:

- `...-ESOptimizer/` — one value per generation: `Fitness_over_generations.csv`,
  `ES_search_mean.csv`, `Fitness_vs_fixed_quota.csv` (one series per candidate,
  parameter value on x), `Mean_candidate_fitness_1_std.csv`;
- `...-RayOptimizer/` — one value per training iteration: `Train_reward.csv`,
  `Training_timing.csv`, ...;
- `...-regulated_env_...|mode=train|ps=...|ss=.../` — one value per environment
  step of the last episode: `Fish_biomass.csv`, `Realized_harvest.csv`,
  `Reward_all_agents.csv`, `Mean_reward_across_agents_1_std.csv`, ...

```python
import pandas as pd
df = pd.read_csv("results/bilevel/<world>-ESOptimizer/Fitness_over_generations.csv")
df.pivot(index="x", columns="series", values="value")
```

Without `--reporter csv`, the same queries are logged as Plotly figures to a
W&B run (offline here: `wandb sync` uploads it later).

## 5. Where the queries come from

`examples/bilevel_fishery/queries.py` holds the query bundles attached at the
three levels in `debug.py::build_config`:

- the regulated environment: `.environment(schema=FisheryMetricSchema, queries=...)`;
- the inner optimizer: `.reporting(schema=RaySchema, queries=...)`;
- the outer optimizer: `.reporting(schema=ESSchema, queries=...)`.

A query names an x path and y paths in the schema; `"*"` matches every runtime
id (agent, policy, candidate) and `reduce="mean", error="std"` averages across
them. `tutorials/visualization.ipynb` builds loggers and queries by hand on
synthetic data and is the place to learn the mechanics.

## 6. Scale up

Drop the flags to get the reference configuration (10 fishers, 4 candidates,
50 APPO iterations per candidate, 1000 generations, 3 evaluation seeds) and a
real W&B project (`--project`). Per-policy learner queries stay empty until
APPO has collected its first four batches.
