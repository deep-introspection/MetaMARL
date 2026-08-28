# Merge notes for Nadine

This file records what was done on the two testing branches while you were
away, the decisions we had to take on your behalf, and the order in which we
suggest merging. Everything below was verified by tests and by short end-to-end
runs; nothing was pushed to `dev`.

## Branch layout

```
origin/dev ──► chore/cleanup-base ─────┬──► feature/social-influence-testing
                                       └──► feature/logging-testing
```

`chore/cleanup-base` holds the cleanup that both features need and that was
already broken on `dev`: the test suite (it imported the pre-rename `src.*`
package and an older `ESConfig.training(dimension=, pop_size=)` signature, so
nothing ran), the coverage configuration (`--cov=src`), a GitHub Actions
workflow, and the removal of `legacy_code/` and the underscore-prefixed test
scripts. Each `*/testing` branch is your feature branch plus a merge of that
base.

Suggested merge order: `chore/cleanup-base` into `dev` first (no behavior
change), then `feature/social-influence-testing`, then
`feature/logging-testing` (see the conflict map below).

## feature/social-influence-testing

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

## feature/logging-testing

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

## Code review findings on the shared core (from the docstring pass, 2026-08-27)

Found while documenting `core/world`, `core/adaptors/ray`, `core/optimizers/{base,config}.py`,
`core/callbacks.py`, `core/utils.py`, `core/annotations.py`. Stated in the docstrings, not
fixed — several touch design choices that are yours.

1. `core/annotations.override` never binds its `OverrideCheck` descriptor (it returns the
   method), so the "subclass of parent" check is dead code; only the name check runs.
2. `World.get_mechanism_by_id` returns `None` on every call after the first successful fetch
   (annotated `MechanismContext`) and raises `TypeError` for a `mode` other than
   `train`/`eval` (`in None`).
3. `World.get_mechanism_by_index` / `get_mechanism_registry`: the registry is keyed by the
   context-id string, not by the candidate index; an int lookup raises `KeyError`.
4. `World.append_context` always overwrites `ctx.id` with a fresh UUID, so its duplicate-id
   check is unreachable; unlike `set_new_context` it does not require `env_id` on mechanisms.
5. `World.flush` / `flush_ctx` each clear one registry only; `_opt_ctx_map` is never pruned,
   so `get_opt_ctx_ids` can return dangling ids.
6. `RayOptimizer._get_policy_handle` references `self.algo`, which does not exist on
   `RayOptimizer` (dead code, would raise `AttributeError`).
7. `RayOptimizer.__init__` divides by `evaluation_config["rollout_fragment_length"]`
   (`TypeError` if unset); `batch_capacity` raises `ZeroDivisionError` with no seeds; `save`
   is a stub returning `None` despite its `_TrainingResult` annotation.
8. `RayOptimizerConfig.debugging` only scales `num_envs_per_env_runner` if `env_runners()` was
   called before it (order-dependent); `evaluation` is annotated `-> None` but returns
   `self`; `rllib_config_mutator` lacks `@staticmethod`; `build_optimizer` leaves
   `opt_id`/`agents` unbound when `world is None` or no `agent_specs` (`NameError` in
   `env_creator`).
9. `PolicyActor.reset` rebuilds a new `Algorithm` without stopping the previous one — the
   July 2026 slowdown (`fix(ray): stop old APPO algorithm before per-generation rebuild` on
   `exp/weekend-variants`) addresses exactly this; not yet on `dev`.
10. `RayRuntimeConfig.disable_cuda=True` by default hides GPUs even with `device="cuda"`;
    `RayRuntime._initialized` is set but never read; `local_mode=True` is hard-coded.
11. `get_policy_loss_if_present` reads only the old-stack path, so on the new API stack it
    always returns NaN — the `policy_loss=NA` in the training log is structural.
12. `Optimizer.batch_capacity` returns `self._batch_capacity`, which the base class never
    sets; `Optimizer.__init__` dereferences `config.env` although `config` defaults to `None`.
13. `OptimizerConfig.environment` resets `env_config = {}` on each call (discarding earlier
    `_merge_env_config`), and its docstring documents RLlib parameters the signature does not
    accept.
14. `EnvStepContext.observation_map` is typed `list[str]` while `BaseEnv.obs_map` is a dict;
    `EnvStepContext.env_id` is `Optional[int]` while `MechanismContext.env_id` is
    `Optional[str]`.
15. `tag_episode_with_env_idx` shadows its `env` parameter immediately and reads
    `policy_seed` without the None-check applied to `seed`/`mechanism_id`.
16. `MechanismStatus.init` is never assigned anywhere.

## Findings from the unit-test pass on the whole core (2026-08-28)

Found while bringing `core/` from 46 % to 97 % coverage without a Ray runtime (the actors
are instantiated through `X.__ray_metadata__.modified_class`). Each point is pinned by a
test that documents the current behaviour, so changing it will make a test fail on
purpose. Nothing was fixed. Points already listed above are not repeated.

17. `core/adaptors/ray/utils.py` uses `to_float(a) or to_float(b)`, so a legitimate value
    of `0.0` is treated as missing and the lookup falls through to the next key
    (`get_episode_return_mean`, `get_env_steps`).
18. `RayOptimizerConfig.build_optimizer` calls `self.copy(copy_frozen=True)`, but on this
    class `freeze` is the deferred RLlib mutator: it records a `freeze` op in `_cfg_ops` and
    leaves `_is_frozen` at `False`. The optimizer therefore receives a mutable config, and
    the recorded `freeze` op would be replayed on the `AlgorithmConfig` later.
19. `build_optimizer`'s `env_creator` reads `mode` from the env context but only writes
    `seed` and `policy_seed`, so training environments never receive a `mode` kwarg (only
    evaluation ones do, through `evaluation_config.env_config`).
20. `RayOptimizerConfig._apply_agents_to_rllib` with `seeds=None` raises `TypeError` on the
    `for seed in self.seeds` loop; the "null seed" case only works with `seeds=[None]`, which
    yields module ids ending in `_sNone`.
21. `WandbReporter.plot_ray_result` accepts `log_raw_rllib_episode_metrics` but forwards a
    hard-coded `True`; the parameter is dead.
22. `plot_env_reduced` accepts `reducers` / `ReductionSpec.fn` but never uses them; the
    reduction is entirely driven by the hard-coded `KEEP_METRICS` allowlist.
    `env_reduced._mean_agent_values` has no caller.
23. `es_population._make_parallel_coordinates_figure` carries a hard-coded `display_names`
    map for the water-project parameters (`fixed_quota`, `fine_amount`, …).
24. Dead modules, never imported anywhere: `core/reporting/utils/ray_old_api_stack.py`
    (789 lines), `core/reporting/base.py` (broken import `from torch import Type`),
    `core/adaptors/ray/mps_model.py`, plus `RayOptimizer._build_agent_policy_map` and the
    old-API-stack branch of `callbacks._evaluate_with_fixed_duration_once`. They sit in
    namespace packages (no `__init__.py`), so coverage does not even count them; counting
    them would bring the 97 % figure to roughly 82 %. Deleting them is pending Rémy's call.

## Conflict map between the two features

A trial merge of `feature/logging-testing` into `feature/social-influence-testing`
(2026-08-27 evening, discarded) produced 30 conflicting files, in four groups:

1. **Delete vs modify, resolve by deleting** — the logging branch deleted
   `core/reporting/utils/*.py` (5 files) and moved the fishery variants to
   `deprecated/` (since deleted), while the mechanism branch only reformatted
   them: `core/reporting/utils/{env_reduced,env_step_context,es_population,
   ray_new_api_stack,ray_old_api_stack}.py`, `examples/bilevel_fishery/
   {bilevel,mechanism_v1,regulated_env_shaefer}.py`. Conversely the mechanism
   branch deleted `core/envs/regulated.py`, which the logging branch modified
   (logger pushes): delete it, its logging moved into `marl_regulated.py`.
2. **Documents to concatenate** — `README.md` (one section per feature),
   `QUICKSTART.md` (two short runs), `TODO.md` (two status sections),
   `docs/MERGE_NOTES.md`, `.gitignore`, `examples/registry.py`.
3. **Shared core, real merge work** — `core/envs/base.py`,
   `core/envs/marl_regulated.py`, `core/envs/regulator.py`,
   `core/optimizers/{bilevel,config}.py`, `core/optimizers/es/optimizer.py`,
   `core/adaptors/ray/{optimizer,optimizer_config,utils}.py`,
   `core/callbacks.py`, `core/reporting/wandb.py`. Rule of thumb: keep the
   mechanism branch's control flow (step pipeline, mechanism template, `mechanism=`
   builder) and re-insert the logging branch's additions (`logger`/`reporter`
   attributes, `_log(...)` calls after `actions`, `rewards`, `obs`, the
   `reporting(schema=, queries=)` builders, `_to_logger_payload` in both
   optimizers, `report_metrics()` in `RegulatorEnv._step`).
4. **Examples and tests** — `examples/bilevel_fishery/{debug,regulated_env,
   regulator_env}.py` (the mechanism branch's fishery env has no `infos`-based
   metrics; the logging branch's `FisheryRegulatedEnv` pushes to the logger —
   port the `_log` calls into the hook methods of the new env),
   `tests/conftest.py` (union of both fixture sets).

71 further files merge automatically. Expect the integration merge to take a
focused day; the unit suites of both branches are the acceptance criterion.
