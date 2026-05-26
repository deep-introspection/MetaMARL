# bilevel-fishery

Research framework for **bilevel optimization** applied to sustainable
fishery management.

- **Outer level**: Evolution Strategies optimize the parameters of a
  regulation mechanism (quota, fine, minimum stock threshold).
- **Inner level**: multi-agent RL (PPO via Ray RLlib) trains fisher
  behaviours under the imposed mechanism.

> **Status**: pedagogical rebuild in progress on the
> `rebuild/from-scratch` branch. The historical codebase is archived under
> the `pre-rebuild-2026-05-25` tag (see `docs/bricks/` for progress).

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
make install
```

## Tests & quality gates

```bash
make test           # pytest + coverage
make lint           # ruff check + format check
make format         # auto-fix
make typecheck      # mypy strict
make notebook-test  # execute every notebook end-to-end
make clean          # remove caches
```

## Repository layout

```
src/bilevel_fishery/   # source package
tests/                 # pytest tests
docs/bricks/           # per-brick pedagogical docs
docs/decisions/        # architecture decision records (ADRs)
notebooks/             # per-brick pedagogical notebooks
config/                # YAML experiment configs (populated as bricks land)
```

## License

BSD-3-Clause — see `LICENSE`.
