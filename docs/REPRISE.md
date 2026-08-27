# Resume file — bilevel-fishery handoff (Aug 2026)

This file is the single place to resume work from disk. It records where we
are, the decisions already taken, what was measured, and the next step. Update
it before any long break, phase close, or context reset.

## Situation

Nadine left two unintegrated branches, `feature/logging` and
`feature/social-influence`, each with an AI-assisted `TODO.md` and tutorial
notebooks. `dev` is the validated reference and is not modified. Rémy owns the
work for the week of 2026-08-27; Nadine will not merge anything herself.

Branch strategy:

```
origin/dev ──► chore/cleanup-base ─────┬──► feature/logging/testing
                                       └──► feature/social-influence/testing
```

`chore/cleanup-base` carries the cleanup inherited from `dev` that both
features need (dead tests, config, CI, legacy dirs). Each `*/testing` branch
is Nadine's branch plus a merge of the base. Nothing is pushed to `dev`.

Scope decisions (with Rémy, 2026-08-27):

- Coverage target: > 90 % on pure modules (`core/mechanism`, `core/envs`,
  `core/metrics`, `core/reporting`, schemas). Ray adaptors and the `World`
  actor are excluded from the unit figure (`[tool.coverage.run] omit`) and
  covered by integration tests; 90 % on all of `core/` is an end-of-week bonus.
- Social influence: document that only observation augmentation exists (no
  Jaques et al. KL bonus); do not implement the KL term.
- Logging: fix and test the existing stack, then implement `"*"` wildcards and
  grouped mean ± std. ES scatter / parallel coordinates / full CSV and
  TensorBoard reporters are bonus.
- Keep `transformers`, `peft`, `bitsandbytes`, etc. in the dependencies even
  though nothing imports them (likely a planned LLM-policy project); flag only.

## Phase 0 — `chore/cleanup-base` (done 2026-08-27)

Done:

- Unit suite revived (26 tests green). The tests imported the pre-rename
  `src.*` package and used an `ESConfig.training(dimension=, pop_size=)`
  signature that no longer exists. They now build `ESOptimizer` directly, set
  the population through `batch_capacity`, and drive
  `_sample_population`/`_update_parameters`. The ES search dynamics assertions
  are unchanged and all pass: Nadine's ES implementation is sound.
- `tests/conftest.py`: `FakeWorld` (in-memory stand-in for the `World` actor,
  `ray.get` patched to pass-through) for unit tests; `ray_session` and a
  `FakeReporter` Ray actor for integration tests.
- Integration: `test_es_regulator_loop` rewritten against the real `World`
  actor. The three RLlib-based tests were written against the old
  mechanism-space API and are skipped with an explicit reason; they are
  replaced by per-feature smoke tests in Phases 1–2.
- Removed: `legacy_code/` (both feature branches already delete it), the
  `_`-prefixed test graveyard and its orphan YAML fixtures, and four `core/`
  modules with zero importers and broken imports (`core/loggers/schemas.py`,
  `core/optimizers/ppo/schema.py`, `core/reporting/utils/{visualization_mechanism,bilevel_viz_reporter}.py`).
- Config: `pytest.ini` no longer covers the nonexistent `src`; coverage config
  in `pyproject.toml`; dev tooling moved to a `[dependency-groups] dev` group
  (`pytest`, `pytest-cov`, `ruff`, `nbconvert`, `ipykernel`); `black/isort/
  mypy/flake8` dropped (never configured). `.github/workflows/ci.yml` runs
  lint, unit (with coverage), and integration jobs.

Findings on `dev` worth knowing (not fixed on the base, flagged for the
feature branches or for Nadine):

- `ESConfig.training(**kwargs)` silently swallows unknown keywords. A typo in a
  hyperparameter name is dropped without error.
- `OptimizerConfig.build_optimizer()` references `opt_id` before assignment
  when `world is None` (`NameError`); building without a World is impossible.
- `ESOptimizer.run()` hard-depends on a reporting actor
  (`self.reporting.plot_es_population.remote`) — W&B is not optional.
- `RayOptimizerConfig.freeze()` is an RLlib mutator: it records the RLlib-side
  freeze and does not freeze the Python config object (base `OptimizerConfig`
  does). Tested as observed behavior; design question for Nadine.
- `RegulatedEnv.mechanism_id` is annotated `str` but `EnvStepContext.mechanism`
  is `Optional[int]`; in practice it is an int index.
- `RegulatedEnv._pre_reset` calls `self._debug_remote(...)`, which is not
  defined anywhere (only reached when the World fetch fails).
- `core/registry.py` imports from `examples/` (library depends on examples);
  wheel builds exclude `examples*` so the built package is broken.
- Ruff on the whole repo (after dropping the `legacy_code` exclusion): ~160
  lint findings, mostly in `examples/`; 45 files need `ruff format`. To be
  cleaned per branch, not on the base.

Waiting on Rémy:

- Move `core/registry.py` under `examples/` (or make registration lazy).
- Delete unreferenced `examples/fresh_water/` forks (`regulated_env_ed_hs-v2.py`,
  `_v3.py`, `_v4_no_quota.py`, `regulated_env_raven.py`) and
  `examples/fresh_water/deprecated/` (still imported by `core/registry.py`).

## Phase 1 — `feature/social-influence/testing` (in progress)

Branch = `origin/feature/social-influence` + merge of `chore/cleanup-base`
(one conflict on `debug.py`, resolved in favor of the feature branch since the
script was rewritten anyway).

Done (commits `cf74dce`, `a821086`, and the integration lot):

- The branch imports and the fishery benchmark runs end-to-end:
  `WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug --outer-iters 2
  --train-iters 2 --num-agents 2 --horizon 20 --num-candidates 2 --num-eval-seeds 1`
  finishes in a few seconds (ES over quota + subsidy, social observation on,
  APPO inner loop, evaluation). Run it as a module (`-m`): as a script,
  `examples` is not importable.
- All six mechanism classes are concrete (`dimension/encode/decode/clip/
  param_names/to_vector`), validated with `ValueError`. Subsidy indexing,
  ParallelMechanism API, `K` NameError, dict/array concatenation, wrong-channel
  dispatch, undefined `_debug_remote`, `@override(MultiAgentEnv)` NameError are
  fixed. `MechanismSpace` is replaced by the mechanism template everywhere
  (`BilevelConfig.mechanism(mechanism=...)`).
- Fishery: 2-component action decomposed (harvest fraction, restoration
  effort), restoration feeds the dynamics through
  `ecology_cfg["restoration_effectiveness"]` (default 0.0, `debug.py` uses 20.0
  — heuristic scale, to confirm with Nadine), intrinsic-reward hook added
  (delivered harvest fraction, same semantics as `dev`).
- Tests: 100 unit tests (`tests/mechanism`, `tests/envs`, `tests/examples`,
  `tests/unit`) + 4 fishery integration tests (env-only and bilevel smoke, with
  and without the social mechanism). Coverage of `core/mechanism` 92–100 %,
  `core/envs/marl_regulated.py` 95 %, `core/envs/regulator.py` 67 %.
- Ray + `uv run` trap fixed in `RayRuntimeConfig.initialize()` (Ray's uv-run
  hook injected a local `working_dir` rejected by `local_mode`). Both
  `uv run python -m ...` and `.venv/bin/python -m ...` work now.
- `TODO.md` updated: dated status section, verified boxes checked.

Open on this branch:

- Tutorial notebooks (`tutorials/mechanism_algorithms.ipynb`,
  `tutorials/custom_benchmark_creation.ipynb`): still 90 % commented-out code;
  to be made executable on the final API and run through nbconvert.
- Docstrings on the modules not touched yet (`core/optimizers/*`,
  `core/adaptors/ray/*`, `core/world/*`), README/AGENTS/QUICKSTART (Phase 3).
- Decisions for Rémy: `violation_transition_width` (quota) was unused and has
  been removed — restore if a violation channel is planned; restoration
  effectiveness default; whether `previous_actions` should be reset to zeros
  or to the last episode's actions at `reset`.
- Coverage of `core/optimizers/es/optimizer.py` (`run()` fixed-mode branch,
  plotting) and `core/envs/regulator.py` decode path under the 90 % target.

Next step: notebooks (Phase 1e), then create `feature/logging/testing`
(Phase 2).

## Commands

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run python -m pytest -m "not integration and not notebook"   # unit + coverage
uv run python -m pytest -m integration --no-cov                   # needs Ray
```
