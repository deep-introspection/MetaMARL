# Working in this repository

This file is for contributors and for coding assistants alike. It says what
the framework is made of, which invariants must hold, how to run and test
things, and which traps have already cost a day.

## What you are looking at

`bilevel-fishery` optimizes a regulation (a *mechanism*) with an outer
Evolution Strategies loop while an inner reinforcement-learning loop (RLlib
APPO) trains the regulated agents. Everything shared between the two levels
goes through one Ray actor, the `World` (`core/world/base.py`): the outer loop
publishes `MechanismContext`s, the inner environments publish
`EnvStepContext`s, and the outer environment reads them back to compute a
fitness. No optimizer or environment holds a reference to another one.

Read `docs/ARCHITECTURE.md` before touching `core/`. Read `docs/REPRISE.md`
first in every session: it is the resume file (where we are, decisions, next
step) and must be updated before a break or a phase change.

## Map

| Path | Role |
| --- | --- |
| `core/optimizers/config.py` | `OptimizerConfig`: fluent, freezable config; `build_optimizer()` instantiates the optimizer and its env |
| `core/optimizers/bilevel.py` | `BilevelConfig` (composition root) and `BilevelOptimizer` (outer loop) |
| `core/optimizers/es/` | Evolution Strategies outer optimizer (`ESConfig`, `ESOptimizer`) |
| `core/optimizers/{appo,ppo}/config.py` | inner optimizer configs (thin subclasses of `RayOptimizerConfig`) |
| `core/adaptors/ray/` | RLlib glue: deferred config ops, one RLModule per (candidate, seed), `PolicyActor`, Ray runtime |
| `core/envs/base.py` | `BaseEnv`: template method publishing every step to the World |
| `core/envs/regulator.py` | `RegulatorEnv`: the outer env (publish candidates, train inner, aggregate) |
| `core/envs/marl_regulated.py` | `MultiAgentRegulatedEnv`: the inner, mechanism-regulated multi-agent env |
| `core/mechanism/` | the mechanism abstraction: a `MechanismSpace` protocol with `encode`/`decode`/`clip`/`sample` and a `VectorMechanism` value object; the social-influence refactor of this package lives on `feature/social-influence-testing` |
| `core/world/` | `World` actor and the context schemas (`MechanismStatus` lifecycle) |
| `core/reporting/` | reporting backends |
| `examples/registry.py` | string-to-class registry for the YAML experiment loaders (`examples/*/bilevel.py`) |
| `core/callbacks.py` | `tag_episode_with_env_idx`: encodes `env|m|ps|ss` in the episode id so RLlib maps episodes to policies |
| `examples/bilevel_fishery/` | reference benchmark: `regulated_env*.py`, `regulator_env.py`, `contexts.py` (fitness), `debug.py` (runnable config) |
| `tests/` | `unit` (no Ray, `FakeWorld` fixture), `integration` (real `World` actor), `notebook` (nbconvert) |

## Invariants

- **Levels talk only through the World.** Never pass an optimizer or an env
  handle across levels; publish a context instead.
- **Mechanisms are immutable values.** Candidates are decoded copies; per-step
  state lives in the environment.
- **One env per (candidate, seed).** `World.get_mechanism_by_id` moves a
  candidate from `published` to `train`/`eval` on its first fetch; a second env
  asking for the same pair gets `None` and steps inertly with zero rewards.
- **Seeds are fixed at construction.** RLlib's per-reset seed is ignored on
  purpose; `policy_seed` identifies the policy trained on an env, `seed` the
  env's own RNG.
- **The ES population size is derived**, never set: it equals the inner
  optimizer's `batch_capacity` (`num_envs_per_env_runner // num_seeds`), so
  every candidate is evaluated by exactly one env per seed. Antithetic ES
  needs an even size unless `break_symmetry=True`.
- **`dev` is not modified directly.** Work on `feature/*` branches; the
  decisions taken on each are logged in `docs/MERGE_NOTES.md`.

## Commands

```bash
uv sync --group dev
uv run ruff check core tests examples/bilevel_fishery examples/cartpole examples/dummy tutorials
uv run ruff format --check .                                          # fresh_water is excluded from lint until it is ported
uv run python -m pytest -m "not integration and not notebook"      # fast, with coverage
uv run python -m pytest -m integration --no-cov                      # starts a local Ray
uv run python -m pytest -m notebook --no-cov                         # executes tutorials/
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug   # the reference run
```

Coverage is measured on all of `core/` with branch coverage and no omit list
(`[tool.coverage.run]` in `pyproject.toml`): the Ray actors are unit-tested
through their unwrapped classes (`X.__ray_metadata__.modified_class`) and the
World through the `FakeWorld` fixture, so no module is excluded. The unit suite
currently reports 99 %; keep it above 90 %.

## Conventions

- Python 3.12, `uv`, `ruff` (lint + format, line length 88). Code, comments,
  docstrings, commit messages and documentation are in English.
- NumPy-style docstrings with array shapes and units; every module has a
  docstring saying what it is for. Every public symbol (a name not starting
  with `_`) in `core/` and `examples/bilevel_fishery/` carries a docstring and
  full type hints (pass of 2026-09-01, commit 080d43c); keep it that way when
  adding one.
- Validation of public parameters raises `ValueError`; `assert` is for
  internal invariants only.
- Tests: TDD when fixing a bug (a failing test first), `pytest` markers
  `unit`/`integration`/`notebook`, no Ray in unit tests (use `FakeWorld` and
  `FakeReporter` from `tests/conftest.py`).
- Commits: `type(scope): imperative summary` (`feat`, `fix`, `test`, `docs`,
  `chore`, `style`), one subject per commit.
- Notebooks are committed without outputs and must execute through
  `tests/notebooks/test_tutorials.py`.

## Traps

- **Run scripts as modules from the repository root**
  (`python -m examples.bilevel_fishery.debug`); as a plain script, `examples`
  is not importable.
- **Ray and `uv run`.** Ray detects a driver started with `uv run` and injects a
  local `working_dir` into the runtime env, which `local_mode` rejects
  (`"... is not a valid URI"`). `RayRuntimeConfig.initialize()` disables that
  hook; if you call `ray.init` yourself, set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
  or use `.venv/bin/python`.
- **Ray runs in `local_mode=True`** (`core/adaptors/ray/runtime.py`), so there
  is no real parallelism despite `num_cpus`; it is deliberate for debugging.
- **Stale bytecode.** After switching branches, remove `__pycache__`
  directories under `core/` and `examples/` before running.
- **`--strict-markers`**: a test with an undeclared marker fails collection;
  declare new markers in `pytest.ini`.
- **`ESConfig.training(**kwargs)` swallows unknown keywords silently**; a
  typo in a hyperparameter name is dropped without error.
- **The reporting backend is Weights & Biases**; use `WANDB_MODE=offline`
  when no account is configured.
- **`.gitignore` ignores `*.csv`, `*.json`, `*.txt` globally**; force-add
  fixtures deliberately.

## Where to look when something is off

- A run hangs or crawls after a few generations: see the July 2026 notes in
  `docs/REPRISE.md` (RLlib `Algorithm` rebuilt every generation, W&B table
  re-rendering).
- Fitness is `-inf` for a candidate: no `EnvStepContext` reached the World
  for that candidate; check the mechanism lifecycle (`MechanismStatus`) and
  the episode tagging callback.
- Observation-shape errors from RLlib: the declared `observation_space` must
  match the benchmark features plus what the mechanism appends.
