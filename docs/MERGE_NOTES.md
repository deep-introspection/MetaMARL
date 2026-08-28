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

Cleanup decided by Rémy on 2026-08-27 and applied to both branches:

- `core/registry.py` moved to `examples/registry.py` (the library no longer
  imports example code; the YAML loaders import it from there). On the
  mechanism branch the fresh-water example is not registered: it still uses
  the pre-mechanism environment API (`@override(MultiAgentRegulatedEnv)` on
  methods the new base does not define) and will need porting.
- Deleted: `examples/fresh_water/deprecated/`, the unreferenced
  `regulated_env_ed_hs-v2/_v3/_v4_no_quota.py` and `regulated_env_raven.py`,
  and (logging branch) `examples/bilevel_fishery/deprecated/`. Git history
  keeps them.
- The live `examples/fresh_water/regulated_env_ed_hs.py` references undefined
  names (`underuse_penalty`, `underuse_severity_m3s`, `stock_shortage_severity`);
  left as is for you.

## feature/logging/testing

State on arrival: the stack imported but could not run — `WandbReporter`
crashed on every `reduce="mean"` query (wrong keyword), the CSV and
TensorBoard reporters implemented an older `_report` signature,
`RayOptimizer.stop()` had a typo, `core/adaptors/ray/protocols.py` imported a
removed package, and `FisheryMetricSchema` could not be instantiated (its
`Optional` fields had no default, so pydantic made them required). Fixed, and
the fishery run works end-to-end with W&B offline or the CSV reporter:

```bash
WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \
    --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20 [--reporter csv]
```

Decisions taken for you:

1. **Reporting is optional everywhere.** `BaseEnv` accepts
   `reporter_cfg=None` and skips logging when no `schema` is given (`_log`
   helper); `Optimizer.report_metrics` is a no-op without a reporter;
   configs build reporters only when one is set. Unit tests and Ray-free
   runs no longer need W&B.
2. **Env-level queries reach the env.** `OptimizerConfig.environment(queries=,
   schema=)` stored `_reporting_queries_env`/`_reporting_schema_env`, but the
   base `build_optimizer` passed the *optimizer*-level schema and queries to
   the env creator (and the `_env` attributes did not exist until
   `environment()` was called). Fixed; the RLlib config already did it right.
3. **Rewards are logged once**, in `MultiAgentRegulatedEnv.step()`, whatever
   path `_step` took. Your branch pushed them both in `step()` and in
   `reward()`; the fishery env bypasses `reward()` so removing the `step()`
   push would have emptied `reward_mean` (it did, during the first smoke run).
4. **`reward()` regression**: the generic `reward()` had become
   `-penalty * violation`, dropping `u_i` relative to `dev`. Restored to
   `u_i - penalty * violation`. Tell us if the change was intentional.
5. **`ESSchema`** gains `generation` (SERIES) and `generation_best`;
   `search_mean`/`global_best`/`generation_best` are keyed by `ParameterName`
   (your TODO §5.1). Queries use `x=("generation",)`.
6. **CSV reporter** writes one long-form file per query
   (`query, x, series, value`; mean/std as two series). **TensorBoard**
   becomes an optional extra (`uv sync --extra tensorboard`) with a lazy
   import; tags are `<title>/<series>`.
7. The query bundles of your TODO §1 live in
   `examples/bilevel_fishery/queries.py` and `debug.py` is an argparse script
   (`--reporter wandb|csv`).

8. **Wildcards and grouping (your TODO §2–§3)** are implemented in the base
   `Reporter`: `"*"` expands at dynamic nodes in sorted order, x and y are
   aligned by their bindings (no Cartesian product), `reduce="mean"` groups
   by the first wildcard when there are two or more levels and averages the
   rest; with a single level it averages across matches. Backends now receive
   labeled `Series(label, x, y, error)` instead of `(x, ys)`, so W&B, CSV and
   TensorBoard only render. `SeedRolloutSchema.aggregate` (§3.1): option B,
   no schema change.
9. Episode-level wildcard queries over `by_episode` (§3, §4) are not usable
   yet: episode ids are unique while the inner logger accumulates per
   training iteration, so those series have length 1 against an `iter` axis
   of length `train_iters`. Aligning them needs an episode-to-iteration key —
   your call.

10. Per-policy learner queries (`train/learner/by_policy/*`) are empty on
    short smoke runs: with `circular_buffer_num_batches=4`, APPO reports no
    `learners` block until four batches have been collected, so the wildcard
    matches nothing and no figure/CSV is written (by design of the resolver:
    zero series -> nothing rendered). They populate on real runs.

Friction observed while writing the visualization tutorial (not changed):

- The configured schema/queries are only reachable through private attributes
  (`_reporting_schema`, `_reporting_queries`, `_reporting_*_env`); a public
  getter would help notebooks and tests.
- `WandbReporter._figure` is the only way to get the Plotly figure without a
  run; a public `figure()` would make offline inspection cleaner.
- `MetricLogger.push_data` at the root requires the exact declared schema type
  (no subclass), while nested nodes accept runtime subtypes.

Known and left as is: `FisheryRegulatorEnv.aggregate_rewards` sets
`mean_fines` to `tail_fish.mean()` (copy-paste; `dev` used the violation
signal). It only feeds `FitnessContext.total_fines`, which the objective does
not use, so the fitness is unaffected — but please fix the intent.

## Conflict map between the two features

Both features modify `core/envs/marl_regulated.py`, `core/envs/base.py`,
`core/optimizers/bilevel.py`, `core/optimizers/es/optimizer.py` and
`examples/bilevel_fishery/debug.py`. The mechanism branch changes the step
pipeline and the constructor of the regulated env; the logging branch adds
`logger`/`reporter` attributes and `push` calls inside the same methods. When
merging the second feature, keep the mechanism branch's pipeline and re-insert
the logging `push` calls after `actions`, `rewards` and `obs` are computed.
