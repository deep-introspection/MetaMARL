# Contributing to bilevel-fishery

## Language

All committed artifacts (code, docs, notebooks, configs, commit messages,
PR descriptions) **must be in English**. Issues and discussions can be
bilingual, but anything that lands in git stays in English so the
international collaborators can read it.

## Commit conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/) with
a scope per brick where applicable:

```
feat(brick-N): short imperative summary

Longer body explaining the why, the design decisions, and the
verifications run. Wrap at 72 columns.

Co-Authored-By: ...
```

Common types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.

## Branch model

- `master` — historical codebase (frozen, archived under tag
  `pre-rebuild-2026-05-25`).
- `rebuild/from-scratch` — current pedagogical rebuild.
- Feature branches: `brick-N-short-name` or `fix/short-description`.

## Quality gates (must pass before merge)

```bash
make lint           # ruff check + format check
make typecheck      # mypy strict on src/
make test           # pytest, coverage threshold enforced
make notebook-test  # every notebook executes end-to-end
```

CI runs the same gates on every push and pull request.

## Architecture decisions

Any non-trivial design choice gets its own ADR under
`docs/decisions/D-NNN-short-name.md`. Use the existing
[D-001](docs/decisions/D-001-reward-function.md) as a template. ADRs include:
context, decision, alternatives considered, rationale, consequences,
implementation pointer, references.

## Per-brick documentation

Each new brick has:

- A pedagogical note under `docs/bricks/NN_short-name.md`
- A runnable notebook under `notebooks/NN_short-name.ipynb`
- Tests under `tests/<area>/` mirroring the source layout
- A single commit with a clear `feat(brick-N): ...` message

## Notebooks

Notebooks are versioned but their outputs are stripped via `nbstripout`
(invoked from `make notebook-test`). If your global git config sets
`core.hooksPath`, `pre-commit install` will not auto-install hooks —
run `make precommit` manually before pushing.

## Dependencies

Runtime deps are added **only when a line of source code actually uses
them**. Anticipated or exploratory deps stay out of `pyproject.toml` until
the code that uses them lands. Each new dep should be justified in the
commit message.
