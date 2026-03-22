# Bilevel Fishery

A small research / experimental codebase for bilevel optimization applied to fishery and water-usage regulation problems.

This repository contains core algorithms, environment implementations, example setups, and legacy code used during research and development. The project supports running example experiments (in `examples/`) and contains reusable components in `core/`, `mechanism/`, `optimizers/`, and `world/`.

## Features

- Bilevel optimization primitives and optimizers (e.g., ES, PPO-related code paths).
- Modular environment and mechanism implementations for fishery and water-usage case studies.
- Example experiment configurations under `examples/` demonstrating how to wire up agents, regulators, and mechanisms.
- Legacy folder with earlier experiments and utilities for reproducibility.

## Quickstart

Prerequisites

- Python 3.12+ is recommended (project contains bytecode for 3.12 / 3.13).
- pip or a conda-based environment.

Optional dependencies

Some examples (and optional analysis tools) depend on external scientific packages. A commonly used optional dependency is RavenPy (used for hydrological modeling in water-usage examples). Install RavenPy with conda:

```bash
conda install -c conda-forge ravenpy
```

If you installed RavenPy this way, you're ready to run the `examples/water_usage` scripts. On macOS/conda, if you run into binary or dependency conflicts, try creating a fresh conda environment first:

```bash
conda create -n bilevel-fishery python=3.12 -y
conda activate bilevel-fishery
conda install -c conda-forge ravenpy
```

Install (editable, development)

```bash
# (optional) using conda
conda create -n bilevel-fishery python=3.12 -y
conda activate bilevel-fishery

# from repo root
pip install -e .
```

Alternatively, to install runtime-only dependencies, consult `pyproject.toml` and your preferred environment manager.

Run an example

From the repository root you can run one of the example scripts. For the bilevel fishery example:

```bash
python examples/bilevel_fishery/main.py
```

Or for the water-usage example:

```bash
python examples/water_usage/main.py
```

Some examples rely on YAML configuration files stored alongside the example (e.g. `examples/bilevel_fishery/config.yaml`). Edit them to change experiment parameters.

## Project layout

- `core/` — core types, registries, utilities used across the codebase.
- `mechanism/` — mechanism definitions and spaces.
- `optimizers/` — optimizer implementations and configuration helpers.
- `world/` — environment/world abstractions.
- `examples/` — runnable example experiments (bilevel_fishery, water_usage).
- `legacy_code/` — older scripts, experiments, and supporting utilities kept for reference.
- `tests/` — unit and integration tests.

## Development

Run tests

```bash
# run the full test suite (fast projects only) from repo root
pytest -q
```

Linting

The project includes `ruff` config. Run ruff to check/fix simple style issues:

```bash
ruff check .
# or to apply fixes
ruff check --fix .
```

Type checking / static analysis

If you use mypy or other tools, refer to the project's configuration and add them to your environment as needed.

Debugging examples

Many example scripts print results and save figures under `results/`. See `examples/*/visualization.py` and example main scripts to understand how outputs are produced.

## Tests and quality gates

- Unit tests and integration tests are under `tests/unit` and `tests/integration`.
- After changing code, run `pytest` and `ruff` as a quick quality gate.

## Contributing

Contributions are welcome. Suggested workflow:

1. Create a topic branch for your change.
2. Add tests for new behavior or bug fixes.
3. Run the test suite locally.
4. Open a pull request with a descriptive title and short rationale.

If your change is large, please open an issue first to discuss the design.

## Reproducibility notes

- Example configurations are YAML files (see `examples/*/config.yaml` and related `config_*.yaml`).
- The `legacy_code/` folder contains older experiment scripts if you need to reproduce past runs.

## License

This project is provided under the license in `LICENSE`.

## Contact

For questions about the code base, open an issue or contact the repo owner.


---

Requirements coverage

- Document RavenPy installation as an optional dependency: Done

If you'd like, I can:

- Add a short `README` to each example folder with example-specific run instructions.
- Add a minimal `Makefile` or `scripts/` to standardize running experiments.
- Extract a `requirements.txt` or lockfile for direct reproducible installs.

Tell me which of the above you'd like next.
