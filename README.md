# bilevel-fishery

A research framework for **bilevel optimization of regulatory mechanisms** in
multi-agent resource systems. An outer optimizer searches the parameters of a
regulation (a quota, a subsidy, a penalty); for every candidate, an inner
reinforcement-learning optimizer trains the agents who live under that
regulation; the outcome of the trained agents becomes the fitness of the
candidate. The reference benchmark is a shared fishery: `N` fishers harvest a
single stock with Pella-Tomlinson growth dynamics, and the regulator looks for
the quota (and other levers) that keeps the stock alive while the fishers keep
earning.

Formally the regulator solves

```
max_theta  F(theta, pi*(theta))       subject to    pi*(theta) = argmax_pi  J(pi; theta)
```

where `theta` are the mechanism parameters, `pi` the agents' policies, `J` the
agents' discounted return under mechanism `theta`, and `F` the regulator's
objective (here a mix of harvest and biomass sustainability). The outer problem
is solved by Evolution Strategies (gradient-free, robust to the noisy inner
solution); the inner problem by APPO (RLlib) with one policy per candidate.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/deep-introspection/bilevel-fishery.git
cd bilevel-fishery
uv sync --group dev          # runtime + test/lint/notebook tooling
```

Weights & Biases is the reporting backend of the reference experiment; set
`WANDB_MODE=offline` to run without an account.

## Running an experiment

Each example ships a runnable script that assembles a `BilevelConfig`:

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug
```

Run scripts as modules (`python -m ...`) from the repository root so that the
`core` and `examples` packages import. See `QUICKSTART.md` for a short
configuration that finishes in about a minute, and the `tutorials/` notebooks
for a guided tour of the concepts.

## Repository layout

```
core/                     the library
  optimizers/             OptimizerConfig, BilevelConfig/BilevelOptimizer, ES (outer), APPO/PPO configs (inner)
  envs/                   BaseEnv, RegulatorEnv (outer env), MultiAgentRegulatedEnv (inner env)
  mechanism/              mechanism abstraction (what the regulator optimizes)
  world/                  the World Ray actor: shared blackboard of contexts between levels
  adaptors/ray/           RLlib glue: RayOptimizer, RayOptimizerConfig, PolicyActor, runtime
  reporting/              reporting backends (Weights & Biases, ...)
  callbacks.py            RLlib callbacks tagging episodes with mechanism and seed identity
examples/
  bilevel_fishery/        the fishery benchmark (regulated env, regulator env, config scripts)
  fresh_water/            a water-allocation benchmark (Raven hydrological model)
  registry.py             name-to-class registry for the YAML experiment loaders
  cartpole/, dummy/       minimal sanity examples
tests/                    pytest suite (markers: unit, integration, notebook)
tutorials/                executable notebooks (feature branches)
docs/                     ARCHITECTURE.md, REPRISE.md (resume file), MERGE_NOTES.md
```

## How the pieces fit

1. `BilevelConfig.build_optimizer()` starts Ray, creates the `World` actor,
   builds the inner optimizer (`RayOptimizer` wrapping an RLlib `Algorithm`
   inside a `PolicyActor`) and the outer optimizer (`ESOptimizer` driving a
   `RegulatorEnv`), and ties the ES population size to the number of inner
   environments.
2. Each ES generation, `RegulatorEnv.step(population)` decodes the population
   into mechanisms, publishes one `MechanismContext` per (candidate, seed) to the
   World, trains the inner policies for `train_iters` iterations, evaluates
   them, aggregates the environments' step records into one fitness per
   candidate, and flushes the consumed contexts.
3. Each regulated environment fetches its candidate from the World at reset
   (by `mechanism_id` and policy seed), applies the mechanism while stepping,
   and publishes an `EnvStepContext` per step.

`docs/ARCHITECTURE.md` walks through the same flow with the class names and
the invariants to respect when extending the framework.

## Mechanisms

A regulation is a `Mechanism` (`core/mechanism/base.py`) intervening on the
agent/environment loop through three optional channels — `action`
(`a* = M^A(s, a)`), `reward` (`r* = M^R(r, s, a*, s')`) and `observation`
(`o* = M^O(s, o)`) — and living in an optimizer space through
`dimension`/`encode`/`decode`. Concrete mechanisms: `QuotaMechanism` (smooth
cap on the harvest fraction), `SubsidyMechanism` (reward for restoration
effort), `ThresholdPenaltyMechanism` (penalty below a stock threshold) and
`SocialInfluenceMechanism` (peers' previous actions appended to observations).
`ChainedMechanism` and `ParallelMechanism` compose them. Benchmarks declare
their dynamics with the decorators of `core/envs/hooks.py` (`@reset`,
`@action`, `@reward`, `@transition`, `@observation`) on a
`MultiAgentRegulatedEnv` subclass, and `BilevelConfig().mechanism(mechanism=...)`
gives the same mechanism template to both levels. `QUICKSTART.md` runs it and
`tutorials/` teach it; the first half of `TODO.md` tracks what remains.

## Metrics and reporting

Every level logs into a typed `MetricLogger` built from a pydantic
`MetricSchema` (`core/metrics/`): the regulated environment logs per-step
values (`FisheryMetricSchema`), the inner optimizer the RLlib results
(`RaySchema`), the outer optimizer one record per generation (`ESSchema`, whose
`inner` field carries the reduced inner schema). `Query` objects
(`core/reporting/query.py`) select an x path and y paths in those schemas —
with `"*"` for runtime ids, `reduce="mean", error="std"` for grouped
averages and `color=` for a per-point colour path — and a `Reporter` backend
renders the resolved `Series`: Weights & Biases (Plotly figures), CSV
(long-form files) or TensorBoard (scalars, `uv sync --extra tensorboard`).
A `ParallelCoordinatesQuery` selects several axes at once and renders as a
parallel-coordinates figure (W&B) or a wide CSV table.
`examples/bilevel_fishery/queries.py` holds the reference query bundles
(including `es_parameter_fitness_queries`, coloured by generation, and
`es_parallel_coordinates_query`); `QUICKSTART.md` shows the outputs and
the second half of `TODO.md` tracks what remains (episode-level grouping).

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run python -m pytest -m "not integration and not notebook"   # unit tests + coverage
uv run python -m pytest -m integration --no-cov                   # needs a local Ray runtime
uv run python -m pytest -m notebook --no-cov                      # executes the tutorials
```

Continuous integration (`.github/workflows/ci.yml`) runs the lint, unit and
integration jobs on every push. `AGENTS.md` documents the conventions and the
traps for contributors and coding assistants.

## Branches

`dev` is the validated reference. Feature work happens on `feature/*`
branches; `docs/MERGE_NOTES.md` records the decisions taken on each and the
suggested merge order.

## License

BSD-3-Clause. See `LICENSE`.
