# Bilevel Fishery

> A research framework for finding fishing regulations that keep fish populations alive without destroying the livelihoods of fishermen — by teaching artificial fishermen what it costs to break the rules.

## Overview

BilevelFishery uses bilevel optimisation to search for sustainable fishery regulations. An outer loop acts as a regulator, proposing rules (quotas, fines, stock thresholds). An inner loop trains fishing agents under those rules using multi-agent reinforcement learning. The system iterates until it finds the regulatory mechanism that best balances ecological sustainability with economic welfare.

The target audience is researchers in environmental policy, ecological economics, or computational social science who want to explore how regulation design affects collective behaviour in a shared-resource setting.

## How it works

### The two-loop structure

Bilevel optimisation means there are two nested optimisation problems running simultaneously.

The **outer loop** is the regulator. It uses Evolution Strategies (ES) — a population-based search algorithm — to explore a space of regulatory mechanisms. Each mechanism is a vector of parameters: a fishing quota, a fine amount, a minimum stock threshold, and so on.

The **inner loop** is the fishermen. Given a fixed mechanism, N fishing agents learn (via reinforcement learning, specifically APPO) how to harvest fish to maximise their income while avoiding penalties. They do not cooperate — each agent acts independently and selfishly.

The outer loop uses the agents' collective behaviour as a signal: mechanisms that lead to stock collapse or low welfare are penalised, and mechanisms that sustain the fishery are rewarded.

```
ES outer loop
  |  proposes mechanism (quota, fine, stock threshold...)
  v
RL inner loop
  |  N agents fish under this mechanism for T steps
  v
Fitness signal
  |  sustainability + welfare -> back to ES
  v
ES outer loop  (next iteration)
```

The cycle repeats for `outer_iters` ES generations. Each ES generation triggers `train_iters` full APPO training iterations in the inner loop.

### The ecology model

Fish and algae populations co-evolve according to Lotka-Volterra predator-prey dynamics (Euler integration at each environment step):

```
dX/dt = delta * X * Y  -  gamma * X  -  H(t)     (fish grow on algae, die naturally, get harvested)
dY/dt = alpha * Y  -  beta * Y * X                (algae grow, get consumed by fish)
```

where `X` is fish biomass, `Y` is algae biomass, and `H(t)` is total harvest at time `t`. The parameters `alpha`, `beta`, `delta`, `gamma` are set in the `ecology_cfg` section of the config file.

### The fitness function

After each inner-loop run, the regulator scores each mechanism candidate:

```
fitness = mean_reward  -  sustainability_weight * sustainability_penalty
```

where `sustainability_penalty` is the average normalised shortfall of the fish stock below `sus_threshold`. A mechanism that produces high harvests but collapses the fishery will score poorly.

## Installation

Install with `uv` (recommended):

```bash
git clone <repo-url>
cd BilevelFishery
uv sync
```

Or with pip in editable mode:

```bash
pip install -e .
```

Ray is a required dependency and is installed automatically. Do not use `uv run` to launch experiments — it conflicts with Ray's internal environment variable setup.

## Quickstart

Run the default bilevel experiment (V0 mechanism, PPO inner loop):

```bash
./run.sh
```

Run the V1 experiment with APPO and risk-sensitive penalties:

```bash
./run.sh examples/bilevel_fishery/main_appo_one_mechanism_v1.yaml
```

Both commands activate the virtual environment automatically and call the correct Python entry point. You must run `uv sync` at least once before using `./run.sh`.

To change experiment parameters, edit the YAML config file passed to `./run.sh`. See `examples/bilevel_fishery/config.yaml` for the default configuration and inline comments explaining every parameter.

## Notebooks

For a guided introduction to the experiment, open the sandbox tutorial notebook:

```
examples/bilevel_fishery/sandbox_tutorial.ipynb
```

This notebook is the best starting point for non-technical users. It walks through the ecology model, the mechanism concept, and what the training curves mean, without requiring any knowledge of reinforcement learning or ES.

## Project structure

```
BilevelFishery/
├── core/                   # Core types, registries, optimizers, environment base classes
│   ├── optimizers/         # ES and APPO optimizer configurations and runners
│   ├── envs/               # Base environment classes (regulator, regulated, MARL)
│   ├── mechanism/          # Mechanism base class and space interface
│   └── world/              # World abstraction and context/event system
├── examples/
│   └── bilevel_fishery/    # Main fishery experiment (see examples/bilevel_fishery/README.md)
│       ├── mechanism.py        # V0 mechanism: quota, fine, ban
│       ├── mechanism_v1.py     # V1 mechanism: risk-sensitive continuous penalty
│       ├── regulated_env.py    # V0 inner-loop environment (fishing agents)
│       ├── regulated_env_v1.py # V1 inner-loop environment
│       ├── regulator_env.py    # Outer-loop environment (fitness aggregation)
│       ├── config.yaml         # Default experiment config (annotated)
│       ├── main_appo_one_mechanism_v1.yaml   # V1 experiment config
│       └── sandbox_tutorial.ipynb            # Interactive tutorial
├── legacy_code/            # Older scripts kept for reference
├── tests/                  # Unit and integration tests
├── run.sh                  # Primary entry point for running experiments
└── pyproject.toml
```

## Data and outputs

Experiments log metrics to Weights & Biases (wandb) when `reporting.reporter: wandb` is set in the config. Key logged quantities include:

- `mean_reward` — average agent reward per mechanism candidate per ES iteration
- `collapse_rate` — fraction of environment steps where fish stock fell below the sustainability threshold
- `mean_fish` — average normalised fish stock over the evaluation window
- `objective_score` — the combined fitness value used by ES

Trajectory plots (fish and algae population over time) are saved under `results/` when an `output_dir` is specified.

## Reproducibility

Set the `experiment.seed` field in the config file and use a fixed `inner.environment.env_config.seed` to reproduce individual runs. The config YAML files in `examples/bilevel_fishery/` capture the full hyperparameter state for each named experiment.

## Development

Run the test suite:

```bash
uv run pytest -q
```

Lint:

```bash
uv run ruff check .
```

Format check:

```bash
uv run ruff format --check .
```

After changing core code, always run `pytest` and `ruff check` before committing.

## Contributing

1. Create a topic branch for your change.
2. Add tests for new behaviour or bug fixes.
3. Run the test suite locally.
4. Open a pull request with a descriptive title and short rationale.

For large changes, open an issue first to discuss the design.

## Optional dependencies

The `examples/water_usage` experiment requires RavenPy for hydrological modelling. Install it with conda:

```bash
conda install -c conda-forge ravenpy
```

On macOS, if you encounter binary conflicts, create a fresh conda environment first:

```bash
conda create -n bilevel-fishery python=3.12 -y
conda activate bilevel-fishery
conda install -c conda-forge ravenpy
```

## License

This project is provided under the license in `LICENSE`.

## Contact

For questions about the codebase, open an issue or contact remy.ramadour@ppsp.team.
