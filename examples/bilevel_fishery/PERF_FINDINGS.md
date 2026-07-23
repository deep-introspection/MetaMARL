# Performance findings — quadratic slowdown of the bilevel run

**TL;DR.** A full run gets slower and slower: per inner-iteration wall time grew
from ~37 s to ~180 s over the first 4 ES generations (~11 h) and kept climbing —
i.e. total time is **quadratic**, so 1000 generations is effectively unreachable.
Two *independent* accumulations cause it. Both are measured (not guessed):

1. **`PolicyActor.reset()` rebuilt the whole RLlib Algorithm every generation** →
   leaks ~20 MB/gen, unbounded. **Fixed** on branch `fix/bilevel-fishery-debug`.
2. **`env_reduced` reporting re-plots an ever-growing per-metric table every inner
   iteration** → O(iters²) work. **Fixed** by throttling the plot to once per
   generation (on the trained policy). Details below.

Plus a minor, cheap memory leak (`_opt_ctx_map`), documented below (not fixed).

Nothing here was pushed.

---

## How the slowdown was localised (methodology)

Per-inner-iteration wall time was split into `train` (the RLlib step) vs `rest`
(everything else: world bookkeeping + reporting), from the driver log:

```
train (policy_actor.train)  = ~3.0 s  CONSTANT across iterations
rest  (world + reporting)   = 12 s → 22.5 s  and climbing (~+0.35 s/iter)
env_steps_iter              = 1600   CONSTANT  (100 horizon × 16 envs)
```

So the RL work per iteration is constant; the growth is pure overhead in `rest`.
The World actor's per-iter calls were timed too (`get_new_env_step_contexts` =
0.001 s, flat) — ruling the World out as the *time* cost. That leaves reporting.

Repro / probe scripts live in the session scratchpad (not committed).

---

## Cause 1 — Algorithm rebuilt every generation (FIXED)

`core/adaptors/ray/policy_actor.py::reset()` did:

```python
self.algo = self.algo_config.build_algo()   # rebuild a whole new Algorithm
```

`reset()` is called once per ES generation (`regulator.py` → `inner.reset()`).
Rebuilding a fresh RLlib `Algorithm` and dropping the old one **does not free the
old one's resources**: RLlib repopulates global registries / connector state on
each `build_algo()` that survive garbage collection.

**Measured** (isolated loop of `reset()` only, no training, RSS of the PolicyActor
process):

| resets | RSS (rebuild, current) | RSS (`stop()`+rebuild) | RSS (`set_weights`) |
|-------:|-----------------------:|-----------------------:|--------------------:|
| 0      | 578 MB                 | 578 MB                 | 578 MB              |
| 10     | 830 MB                 | 829 MB                 | 667 MB              |
| 20     | **1020 MB** (+442)     | **1006 MB** (+428)     | 751 MB              |
| 33→60  | (still climbing)       | (still climbing)       | **plateau 859 MB**  |

- `stop()` before rebuild **does not help** (verified) → the leak is not in the
  stoppable env-runner/learner resources.
- **`set_weights(self._init_weights)`** (the method's original, commented-out
  intent) removes the leak entirely: RSS plateaus, and reset is ~100× cheaper
  (1.0 s → 0.01 s).

**Fix applied** (branch only): `reset()` now calls `self.algo.set_weights(...)`.

End-to-end check (3 gens): per-iter time no longer jumps at generation
boundaries (gen2 iter1 ≈ gen1 iter5), training runs, fitness finite, and the ES
still anneals (sigma 0.5076 → 0.5041 → 0.4991).

**Method caveat for your review:** `set_weights` restores the initial RLModule
weight snapshot but does **not** reset the learner optimizer state, RNG, or
connector running stats. If a *fully* fresh optimizer per generation is required,
a leak-free rebuild path would be needed instead (e.g. a fresh short-lived
PolicyActor per generation that is killed afterwards). Flagging, not deciding.

---

## Cause 2 — `env_reduced` re-plots a growing table every inner iteration (FIXED)

`core/reporting/utils/env_reduced.py::plot_env_reduced` (called every inner
iteration via `RayOptimizer.run`) keeps a module-global, per-metric table that
**accumulates one row per iteration and is never reset**, then re-renders a plot
from the *entire* history each iteration:

```python
_ENV_REDUCED_ITER_TABLES: dict[tuple[int, str], wandb.Table] = {}   # module global
...
table = _ENV_REDUCED_ITER_TABLES.get(cache_key)      # grows every iteration
table.add_data(...)                                  # +1 row/iter, never cleared
_log_iteration_reduced_shaded_plot(table=table, ...) # re-plots the full table
```

Work per iteration ∝ iterations so far → **O(iters²)** total. This is the
dominant remaining creep (`rest` 12 → 22.5 s). It is forced on by default even
when no reducers are configured (`RayOptimizer.__init__` →
`build_default_fishery_reduction_specs()`), and the surrounding code is marked
"temporary / testing".

`plot_env_reduced` also logs several large `wandb.Table`s (raw_env_steps,
wide, derived, correlation, ...) built from 1600 contexts **every iteration** —
constant per call, but a lot of per-iteration work regardless.

**Fix applied** (branch only): `plot_env_reduced` is now called **once per
generation**, on the last inner iteration (the trained policy), instead of every
iteration. Implementation (minimal, opt-in, backward compatible):

- `RayOptimizer` gained `self._train_iters_hint` (default `None`).
- `RegulatorEnv._step` (base class) sets
  `self.inner._train_iters_hint = self.train_iters` before the inner training loop.
- `RayOptimizer.run` only runs the env_reduced block when
  `_inner_iter >= _train_iters_hint` (last iter). If the hint is `None`, legacy
  per-iteration behaviour is preserved.

**Measured (2 gens × 20 iters, before vs after):**

| | per normal iter | per generation |
|---|---|---|
| before (fix #1 only) | 18 → 32 s, climbing | grows without bound |
| after (throttle) | **~3.5 s, flat** | ~3.5 s × 99 + one ~15–18 s plot iter |

No cumulative growth across generations; the run is now effectively linear
(~6 min/generation at these settings). ES behaviour unchanged (still anneals).

The env_reduced trajectory now has **per-generation** resolution (one point per
generation) rather than per-inner-iteration. If you want a different cadence,
`_train_iters_hint` is the single knob. A small O(gens²) residual remains in the
once-per-generation plot render (the accumulating table still grows ~1 batch/gen),
but it is negligible vs the removed O(iters²).

**Alternative if you prefer:** make `env_reducers` fully opt-in (don't default to
`build_default_fishery_reduction_specs()` in `RayOptimizer.__init__`).

---

## Minor — `_opt_ctx_map` grows unbounded (PROPOSED, low priority)

`core/world/base.py`: `flush_ctx()` pops flushed ids from `self._contexts` (which
stays bounded, verified: constant 3200) but never removes them from
`self._opt_ctx_map[opt_id]`, which grows ~6400 ids/iteration for the whole run
(~640 M entries over 1000 gens). CPU cost per call is negligible (the reducer
cursor keeps the slice O(new)), so this is a **memory** leak, not the time leak.

**Proposed fix**: in `flush_ctx`, also drop the already-consumed prefix of each
`_opt_ctx_map[opt_id]` (up to the reducer cursor) and rebase the cursor — this is
safe because those ids have already been returned by `get_new_env_step_contexts`
and their contexts are being flushed. Not applied (touches the reducer cursor
semantics — your call).

---

## Status

- Branch `fix/bilevel-fishery-debug`, **not pushed**.
- Fixed + committed: cause 1 (`set_weights`) and cause 2 (env_reduced throttle).
- Cause 3 (`_opt_ctx_map`): documented here for review, **not** modified
  (memory-only, negligible CPU).
