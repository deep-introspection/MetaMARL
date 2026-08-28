# Resume file — bilevel-fishery handoff week (Nadine away, 2026-08-27 → ~09-05)

**Read this first in every session.** It is the status board of the week: what is
done, what is in progress, what remains, what waits on a decision, and how to
resume from disk. Update it before every `/clear`, phase change or break.

Plan of record: `~/.claude/plans/yo-voici-le-delegated-crescent.md` (Rémy's
machine). Notes for Nadine: `docs/MERGE_NOTES.md`. Contributor guide: `AGENTS.md`.

## Where things live

| Branch | Worktree | Role |
| --- | --- | --- |
| `chore/cleanup-base` | `../bilevel-fishery-base` | shared cleanup + branch-neutral docs; merged into both testing branches |
| `feature/social-influence-testing` | `../bilevel-fishery` (main dir) | Nadine's mechanism branch + base |
| `feature/logging-testing` | `../bilevel-fishery-logging` | Nadine's metrics/reporting branch + base |

`dev` is never modified. Nothing is pushed yet (Rémy decides when).
Run `git merge` **inside the target worktree** — a `cd` chain in one shell
command merged the base into itself twice on day 1.

## Status board

| Phase | Status | Evidence |
| --- | --- | --- |
| 0. Shared base: tests revived, coverage/CI config, dead code removed, README/AGENTS/ARCHITECTURE, docstring pass on shared core | **done** | 26 base tests green; CI workflow; 100 docstrings on 13 files |
| 1. Mechanism branch runnable, tested, documented | **done** | full `pytest`: 154 passed, 3 skipped (unit + integration + 2 notebooks); coverage 92–100 % on `core/mechanism`, `core/envs`; 93–97 % on `core/optimizers/{es,config,base}` |
| 2. Logging branch runnable, tested, wildcards + grouped mean/std, CSV/TensorBoard | **done** | full `pytest`: 130 passed, 3 skipped (unit + integration + notebook); coverage 86 % overall, 97 % on the ES |
| 3. Documentation (README, AGENTS, ARCHITECTURE, QUICKSTART per branch, MERGE_NOTES, TODO status) | **done** | files present on both branches; notebooks execute under `tests/notebooks` |
| 4. Closing: push branches, hand `docs/MERGE_NOTES.md` to Nadine | **waiting on Rémy** | push not requested yet |
| Bonus A. 90 % coverage on all of `core/` (World, Ray adaptors need mocks) | not started | `bilevel.py` 75 %, adaptors/World excluded from the unit figure |
| Bonus B. Logging: ES scatter colored by generation, parallel coordinates, episode-level wildcard alignment | not started | see `TODO.md` §5.4, §5.5, §3–4 on the logging branch |
| Bonus C. Integration trial of the two features | measured, not built | 30 conflicting files, map in `docs/MERGE_NOTES.md` |

## Decisions log

| Date | Decision | By |
| --- | --- | --- |
| 08-27 | Shared cleanup on a base branch merged into both testing branches | Rémy |
| 08-27 | Coverage target > 90 % on pure modules; all of `core/` is a bonus | Rémy |
| 08-27 | Social influence stays observation-only (no Jaques et al. KL bonus) | Rémy |
| 08-27 | Logging scope: fix + test + wildcards + grouped mean/std; ES scatter/parallel-coords bonus | Rémy |
| 08-27 | Keep `transformers/peft/bitsandbytes` dependencies (likely future LLM policies) | Rémy |
| 08-27 | Delete fresh-water forks and every `deprecated/` archive | Rémy |
| 08-27 | Move `core/registry.py` to `examples/registry.py` | Rémy |
| 08-27 | Keep `restoration_effectiveness` 0.0 default / 20.0 in `debug.py` (heuristic, flagged) | Rémy |
| 08-27 | Respect Nadine's ignore of `CLAUDE.md`; `AGENTS.md` is the agent guide | Rémy |
| 08-27 | `BilevelConfig.mechanism(mechanism=...)` is the single builder; mechanism = template | Claude, per TODO |
| 08-27 | Reporters receive labeled `Series`; wildcards group by first level when ≥ 2 levels | Claude, per TODO |

## Waiting on

- **Rémy**: push the three branches? (outward-facing, not done without a go).
- **Rémy**: delete the ~900 lines of dead code listed in `docs/MERGE_NOTES.md` §24 (otherwise the
  honest coverage figure is ~82 %, not 97 %).
- **Nadine** (in `docs/MERGE_NOTES.md`): `mean_fines = tail_fish.mean()` intent; undefined
  names in the live `examples/fresh_water/regulated_env_ed_hs.py`; the 16 code-review
  findings; episode-level wildcard alignment; the fresh-water example is not ported to the
  mechanism API.

## Known traps (details in `AGENTS.md`)

Run scripts as modules (`python -m examples.bilevel_fishery.debug`); Ray + `uv run` is
handled in `RayRuntimeConfig`; clear `__pycache__` after switching branches; never
`git commit --amend` before the suite is green; `WANDB_MODE=offline` for runs.

## Resume commands

```bash
# any worktree
uv sync --group dev
uv run ruff check core tests examples/bilevel_fishery examples/cartpole examples/dummy examples/registry.py tutorials
uv run python -m pytest -m "not integration and not notebook"      # unit + coverage
WANDB_MODE=offline uv run python -m pytest                          # everything (~1–2 min)
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20
```

## Day log

- **2026-08-27** — Phases 0–3 completed on both branches (see status board). Smoke runs
  of both `debug.py` scripts pass end-to-end. Cleanup decisions applied. Docstring pass
  merged. Trial merge of the two features measured.
