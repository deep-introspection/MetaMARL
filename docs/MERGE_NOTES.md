# Merge notes for Nadine

This file records what was done on the two testing branches while you were
away, the decisions we had to take on your behalf, and the order in which we
suggest merging. Everything below was verified by tests and by short end-to-end
runs; nothing was pushed to `dev`.

## Branch layout

```
origin/dev ──► chore/cleanup-base ─────┬──► feature/social-influence/testing
                                       └──► feature/logging/testing
```

`chore/cleanup-base` holds the cleanup that both features need and that was
already broken on `dev`: the test suite (it imported the pre-rename `src.*`
package and an older `ESConfig.training(dimension=, pop_size=)` signature, so
nothing ran), the coverage configuration (`--cov=src`), a GitHub Actions
workflow, and the removal of `legacy_code/` and the underscore-prefixed test
scripts. Each `*/testing` branch is your feature branch plus a merge of that
base.

Suggested merge order: `chore/cleanup-base` into `dev` first (no behavior
change), then `feature/social-influence/testing`, then
`feature/logging/testing` (see the conflict map below).

## feature/social-influence/testing

State on arrival: the branch did not import (`@override(MultiAgentEnv)` on
methods `MultiAgentEnv` does not define, imports of the deleted
`core/mechanism/space.py`, no `Mechanism` subclass instantiable). The P0
items of your `TODO.md` are now done and the fishery benchmark runs
end-to-end:

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20
```

Decisions taken for you (all reversible, all flagged in code comments):

1. **Single builder signature**: `BilevelConfig.mechanism(mechanism=...)`. The
   mechanism instance is the *template*: it defines the optimizer space
   (`dimension`, `encode`/`decode`) and is the default mechanism of the
   regulated envs until a candidate is published. `MechanismSpace` is gone
   from `BaseEnv`, `RegulatorEnv`, `ESOptimizer` and `BilevelConfig`
   (`env.mechanism_template` replaces `env.m_space`).
2. **What is optimized**: `QuotaMechanism.fixed_quota` and
   `SubsidyMechanism.subsidy` (normalized by `MAX_SUBSIDY = 0.5`) are the two
   optimized parameters; `ThresholdPenaltyMechanism` and
   `SocialInfluenceMechanism` are fixed (`dimension == 0`). This matches what
   `debug.py` optimized before.
3. **Social influence is observation augmentation only.** The Jaques et al.
   KL reward bonus is not implemented; `influence_weight` is kept as a
   reserved, unused field and the module docstring says so.
4. **Observation layout**: `[benchmark features, mechanism.to_vector(),
   quota allowed_frac, peers' previous actions]`. The quota appends
   `allowed_frac` from `reset` onward (computed from the resource level) so the
   observation size is constant; `debug.py::observation_dim` computes it.
5. **Restoration enters the ecology** through
   `ecology_cfg["restoration_effectiveness"]` (biomass per unit of total
   effort; default `0.0`, `debug.py` uses `20.0` as a heuristic scale — please
   confirm or replace). Incentives stay in `SubsidyMechanism`.
6. **Intrinsic reward** of the fishery is the delivered harvest fraction
   (post-quota), the same quantity `dev` used (`delivered / full_required`).
7. **Hooks**: declaring two hooks of the same type on one class raises
   `TypeError` at class definition instead of the last one silently winning.
8. `QuotaMechanism.violation_transition_width` was validated but never used
   and was removed; restore it if a violation channel is planned.
9. `RayRuntimeConfig.initialize()` disables Ray's `uv run` runtime-env hook,
   which injected a local `working_dir` that `local_mode` rejects. This is why
   `uv run python ...` used to fail while `.venv/bin/python ...` worked.

Friction observed while writing the tutorials (not changed, for you to decide):

- `World.get_mechanism_by_id` moves a candidate `published -> train` on its
  first fetch, keyed on `(index, seed)`. A second env asking for the same
  `(mechanism_id, policy_seed)` silently gets `None` and steps inertly with
  zero rewards. Correct for one env per (candidate, seed), but a warning or an
  explicit error would save a user an hour.
- `SubsidyMechanism.reward` reads its context from `kwargs["action_after"]`
  (a name chosen by the env) while every other mechanism gets context through
  `bindings`. Two injection paths for the same idea.
- `QuotaMechanism` caches `allowed_frac` in a mutable `_context` on a frozen
  dataclass (your TODO §16); harmless with one instance per env, but the
  "mechanisms are immutable" story is not fully true.
- `RayRuntimeConfig.num_cpus`/`num_gpus` were stored but never passed to
  `ray.init`; fixed on this branch.

Still open on this branch: quota numerical parity against `dev` (TODO §7),
mechanism-local `_context` statefulness (TODO §16 — one `QuotaMechanism`
instance must not be shared by concurrently stepping envs; today each env gets
its own decoded candidate, so this holds), context publishing seed tests
(TODO §13).

## feature/logging/testing

(To be completed in Phase 2.)

## Conflict map between the two features

Both features modify `core/envs/marl_regulated.py`, `core/envs/base.py`,
`core/optimizers/bilevel.py`, `core/optimizers/es/optimizer.py` and
`examples/bilevel_fishery/debug.py`. The mechanism branch changes the step
pipeline and the constructor of the regulated env; the logging branch adds
`logger`/`reporter` attributes and `push` calls inside the same methods. When
merging the second feature, keep the mechanism branch's pipeline and re-insert
the logging `push` calls after `actions`, `rewards` and `obs` are computed.
