# Quickstart — from a clone to a first run in ten minutes

This walkthrough runs the fishery benchmark end-to-end on a laptop, without a
Weights & Biases account, with the two features this branch integrates: the
explicit mechanism abstraction (quota, restoration subsidy and social
observation) and the typed metrics/reporting stack (schemas, queries and the
W&B, CSV and TensorBoard reporters).

## 1. Install

```bash
git clone https://github.com/deep-introspection/bilevel-fishery.git
cd bilevel-fishery
git checkout feature/integration-trial
uv sync --group dev
```

## 2. Check the toolchain

```bash
uv run python -m pytest -m "not integration and not notebook"
```

The unit suites of both features run in a few seconds, with a coverage table
for `core/`.

## 3. Run a short experiment

The same script serves both features. The first run reports to an offline W&B
run (the default reporter):

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

The second run is the same experiment with the CSV reporter, which needs no
W&B at all and writes every query to disk:

```bash
uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 \
    --num-candidates 2 --num-eval-seeds 1 --reporter csv --output-dir results
```

`--reporter` accepts `wandb` (the default) or `csv`; `--project` names the
reporter project and `--output-dir` the CSV root.

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

## 5. Read the outputs

With `--reporter csv`, every query becomes one long-form CSV under
`results/bilevel/<world>-<owner>/<query title>.csv`, with columns
`query, x, series, value, error, color`:

- `...-ESOptimizer/` — one value per generation: `Fitness_over_generations.csv`,
  `ES_search_mean.csv`, `Fitness_vs_fixed_quota.csv` (one series per candidate,
  parameter value on x, coloured by generation), `Mean_candidate_fitness_1_std.csv`,
  and the wide parallel-coordinates table;
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

With the default reporter, the same queries are logged as Plotly figures to a
W&B run.

## 6. Where the queries come from

`examples/bilevel_fishery/queries.py` holds the query bundles attached at the
three levels in `debug.py::build_config`:

- the regulated environment: `.environment(schema=FisheryMetricSchema, queries=...)`;
- the inner optimizer: `.reporting(schema=RaySchema, queries=...)`;
- the outer optimizer: `.reporting(schema=ESSchema, queries=...)`.

A query names an x path and y paths in the schema; `"*"` matches every runtime
id (agent, policy, candidate) and `reduce="mean", error="std"` averages across
them.

## 7. Learn the mechanics

- `tutorials/mechanism_algorithms.ipynb` — each mechanism by hand, composition,
  a real environment rollout against a `World` actor (about ten seconds).
- `tutorials/custom_benchmark_creation.ipynb` — writing a new benchmark with
  the hook decorators and a custom mechanism, testing the equations before
  any RL.
- `tutorials/visualization.ipynb` — loggers and queries built by hand on
  synthetic data.
- `docs/ARCHITECTURE.md` — the run assembly and the invariants.

## 8. Scale up

Drop the flags to get the reference configuration (10 fishers, 4 candidates,
50 APPO iterations per candidate, 1000 generations, 3 evaluation seeds); use
`--no-social` to drop the social observation and `--project` for a real W&B
project. Per-policy learner queries stay empty until APPO has collected its
first four batches. `TODO.md` lists what remains for each feature (quota
parity against `dev` and per-step mechanism context on the mechanism side,
episode-level grouping on the reporting side).
