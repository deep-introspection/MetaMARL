# Changes — branch `fix/packaging-simplify-install`

Here is a summary of everything that was done on this branch. Two main areas: fixing the install so the project actually runs out of the box, and a full documentation pass on the entire codebase.

---

## 1. Installation & packaging fixes

### The problem
Running `uv sync` then `uv run python -m examples.bilevel_fishery.main` crashed with two errors:

1. `ModuleNotFoundError: No module named 'examples'` — because `pyproject.toml` excluded `examples*` from setuptools package discovery, so the `examples` package was never installed into the virtualenv.

2. `ModuleNotFoundError: No module named 'examples.water_usage'` — `core/registry.py` still imported from `examples.water_usage.*`, which no longer exists (your latest push on `dev` already had this commented out, so it was resolved by pulling).

### What was changed

**`pyproject.toml`** — Removed `examples*` from the setuptools package discovery exclusion list. The `examples` package is now installed as part of `uv sync`, so absolute imports like `from examples.bilevel_fishery.bilevel import ...` work without any `PYTHONPATH` workaround.

```toml
# before
exclude = ["legacy_code*", "tests*", "examples*"]

# after
exclude = ["legacy_code*", "tests*"]
```

**`run.sh`** (new file at repo root) — A simple launcher script so that users never need to type the full `python -m examples.bilevel_fishery.main --config ...` command. It activates the virtualenv directly and calls Python, deliberately avoiding `uv run` to prevent the Ray worker environment variable conflict you mentioned.

```bash
# install once
uv sync

# run (default config)
./run.sh

# run with a specific config
./run.sh examples/bilevel_fishery/main_appo_one_mechanism_v1.yaml
```

---

## 2. Documentation

The codebase had roughly 33% docstring coverage. Every public function and class outside `legacy_code/` and `tests/` has now been documented.

### Python docstrings

All 61 Python files were covered. Format is NumPy-style throughout, consistent with the project's existing conventions. For the most research-facing files, docstrings include:

- **Mathematical equations** — Lotka-Volterra dynamics in `regulated_env.py` and `regulated_env_v1.py`, ES update rules in `core/optimizers/es/optimizer.py`, PPO objective in `core/optimizers/ppo/config.py`, penalty and utility formulae in the fishery environments.
- **Parameter descriptions** — every `__init__` argument documented, including the mechanism space parameters (quota, fine, ban, stock threshold) and their normalisation ranges.
- **Module-level docstrings** on all `main_*.py` scripts explaining which experiment variant each one runs and how to invoke it.

### README files

**Root `README.md`** — Added a *"How it works"* section before the quickstart, with a plain-language explanation of the bilevel optimisation loop and an ASCII flowchart. Fixed the quickstart to use `./run.sh`. Added a *Notebooks* section pointing to `sandbox_tutorial.ipynb`, which was previously invisible to new users.

**`examples/bilevel_fishery/README.md`** (new) — A standalone guide for someone who just wants to run and modify the fishery experiment. Covers: how to run, a full parameter table for `config.yaml` (including the ecology section with the Lotka-Volterra symbols explained), expected W&B outputs, and a side-by-side V0 vs V1 mechanism comparison.

### Config files

Both `config.yaml` and `main_appo_one_mechanism_v1.yaml` now have inline comments on every parameter — units, roles in the differential equations, and guidance on safe ranges to modify.

---

## Files changed

| File | What changed |
|---|---|
| `pyproject.toml` | Removed `examples*` from package exclusion |
| `run.sh` | New launcher script |
| `README.md` | Added bilevel explanation, fixed quickstart, added notebooks section |
| `examples/bilevel_fishery/README.md` | New — full experiment guide |
| `examples/bilevel_fishery/config.yaml` | All parameters annotated inline |
| `examples/bilevel_fishery/main_appo_one_mechanism_v1.yaml` | All parameters annotated inline |
| 61 Python files across `core/` and `examples/` | NumPy docstrings added to all public functions and classes |

No code logic was modified anywhere — only docstrings and packaging configuration.

Rémy
