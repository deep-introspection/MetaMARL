# Brick 0 — Project skeleton

> **Date**: 2026-05-25
> **Branch**: `rebuild/from-scratch`
> **Reference archive**: tag `pre-rebuild-2026-05-25`

## Why this brick

Before writing a single line of domain code, we lay down a **strict tooling
skeleton** that makes everything else easier:

- A single Python config file (`pyproject.toml`) — no separate `ruff.toml`
  or `pytest.ini`.
- A standard `src/` layout that forces tests to run against the installed
  package, not the repo files.
- **Strict** linting from day one (ruff with `E F I B UP RUF SIM N D`); it
  is far cheaper to keep a clean base than to clean it up later.
- Notebooks **kept under version control** but with `nbstripout` to clear
  outputs automatically.
- A minimal CI that runs ruff and pytest on every push.

## Reverse-prompts used (and corrected)

| # | Source prompt | Audit correction |
|---|---|---|
| B0.1 | Init Python 3.12 project + BSD-3 + author | (none) |
| B0.2 | Runtime deps | **Removed** 6 dead LLM deps + moved ruff/pytest out of runtime |
| B0.3 | Dev deps | **Unified** on ruff (vs black+isort+flake8); added nbstripout + pre-commit |
| B0.4 | Setuptools packages | **Explicit `src/` layout** instead of flat |
| B0.5 | `__init__.py` | **Moved** under `src/bilevel_fishery/` |
| B0.6 | Ruff config | **Hardened selection**: `E F I B UP RUF SIM N D` + NumPy docstring convention |
| B0.7 | Pytest config | **Migrated** into `pyproject.toml`, `--cov=src/bilevel_fishery` |
| B0.8 | Gitignore | **`*.ipynb` kept versioned**, cleaned by nbstripout |
| B0.9 | README | **Short and factual** (~50 lines) instead of 145 hallucinated lines |

## Verifications

```bash
make install   # uv sync + pre-commit install (skipped if core.hooksPath set)
make test      # pytest passes (2 tests)
make lint      # ruff check + format check pass
make typecheck # mypy strict passes
```

## What is NOT part of Brick 0

- No domain code (fishers, fish, mechanism)
- No scientific dependencies (numpy, scipy, torch, Ray) — added brick by brick
- No experimental YAML config — `config/` is empty for now

## Next brick

**Brick 1** — Pure ecological model: the predator-prey dynamics (Lotka-Volterra),
without RL or Ray. Plain Python + numpy + scipy.
