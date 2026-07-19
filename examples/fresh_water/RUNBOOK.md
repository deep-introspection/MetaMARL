# Fresh-water bilevel — runnable + convergence fixes

Branch: `fix/fresh-water-runnable`

This branch takes the fresh-water use case from **crashes-on-startup + outer loop
that never converges** to **runs end-to-end + a convergent evolution strategy**.
It is meant to be *read and ported*: every change is small, localized, and
justified below so it can be cherry-picked into the main line.

Entry point audited: `examples/fresh_water/debug.py` → inner env
`regulated_env_ed_hs_v4.py` (APPO), outer env `regulator_env_raven.py` (ES).

---

## TL;DR — what changed and why

| Commit | File(s) | Change | Why it mattered |
|---|---|---|---|
| `fix(fresh-water): make inner env runnable without Raven` | `regulated_env_ed_hs_v4.py`, `regulator_env_raven.py` | Bind every state var *before* the Raven block; Raven reads only overwrite valid values; streamflow-deviation defaults to 0.0 when no baseline | The env crashed (`TypeError`/`UnboundLocalError`) the moment Raven output was missing — the advertised "fallback to internal dynamics" did not exist |
| `fix(es): restore sigma annealing and correct 1/5 success rule` | `core/optimizers/es/optimizer.py` | Success rate on **raw** fitness vs previous best; removed the σ soft-anchor; removed a double `generation++` | σ was pinned near max and never shrank → the outer ES could not converge |
| `refactor(mechanism): stop optimizing inert regulation parameters` | `mechanism.py` | Drop `max_farm_area_m2` and `under_irrigation_penalty_scale` from `optimize_params` | The ES wasted search dimensions on parameters that have **no effect** on fitness |
| `feat(fresh-water): portable, auto-detecting Raven configuration` | `debug.py`, `regulated_env_ed_hs_v4.py` | `RAVEN_CWD`/`RAVEN_CMD`/`USE_RAVEN` env vars; env auto-disables Raven if the model dir/binary is missing | Hard-coded `/Users/nadine/...Raven.exe` broke on every other machine |
| `fix(ray): make local_mode configurable (default False)` | `runtime.py`, `bilevel.py` | `local_mode` is a config field, default `False` | Hard-coded `local_mode=True` crashed `ray.init` on Ray 2.53 from inside the repo |

**Not changed on purpose:** the reward. See [§3](#3-reward--analyzed-no-change-needed).

---

## Does it run?

**Fallback mode (no Raven) — runs anywhere:**

```bash
WANDB_MODE=disabled PYTHONPATH=. uv run python examples/fresh_water/smoke_run.py
# ... -> "CHAIN OK"
```

`smoke_run.py` is a scaled-down mirror of `debug.py` (4 agents, short horizons,
2 outer iterations). Use it as a "does the chain still run?" check after edits.

> ⚠️ **Fallback mode is not physics.** Without Raven the reservoir state simply
> persists (carry-forward); the reward is flat and the regulator has no leverage.
> It exists only to exercise the full pipeline. A scientifically meaningful run
> **requires Raven** (see below). The env logs a loud warning whenever it falls back.

---

## The changes in detail

### 1. Inner env runnable without Raven — `regulated_env_ed_hs_v4.py`

Root cause (reproduced): `_reset` and `transition_kernel` assumed Raven always
returned readings. `transition_kernel` defined `eod_reservoir_stage`, `release_pressure`,
etc. *inside* the `try`/`if use_raven` block, then read them unconditionally in
`new_state = {...}`. Any missing/failed Raven call → `UnboundLocalError` (or
`TypeError: NoneType - float` in `_reset`).

Fix:
- All state variables are now initialized (carry-forward from `S_t` + config
  initial conditions) **before** the Raven block; valid Raven reads (`is not None`)
  overwrite them.
- Raven interaction in `_reset` is wrapped best-effort.
- New fallback config keys (all optional): `fallback_reservoir_stage_m`,
  `fallback_streamflow_m3s`, `fallback_outflow_m3s`, `fallback_precip_mm_day`.
- `_raven_available()` auto-detects the model dir + binary and disables Raven cleanly.

Outer objective (`regulator_env_raven.py`): `aggregate_rewards` no longer crashes
when there is no Raven baseline — `streamflow_deviation` defaults to `0.0` (finite),
so the ES always receives a finite fitness.

Verified: `reset()` + 8 `step()`s run; truncation fires at the horizon.

### 2. ES convergence — `core/optimizers/es/optimizer.py`

Two bugs blocked convergence of the outer loop:

- **1/5 success rule on whitened fitness.** `success_rate = mean(fitness > 0)` was
  computed on the *whitened* fitness (zero mean by construction) → always ≈ 0.5,
  so σ never adapted. Now computed on **raw** fitness vs the previous generation's
  best, so `success_rate → 0` near an optimum and σ anneals down. (Whitening is
  kept for the gradient estimate only.)
- **σ soft-anchor.** `sigma = 0.97*sigma + 0.03*sigma_mid` pulled σ back to mid-range
  every generation, pinning it near the max. Removed.
- Also removed a double `generation += 1`.

Measured on a synthetic optimum (isolated ES probe, no env):

| | σ trajectory | dist to optimum | verdict |
|---|---|---|---|
| before | 0.5 → **0.59 (pinned)** | ~0.067, jittering | did not converge |
| after | 0.5 → **0.05** | ~0.02 | converged |

### 3. Reward — analyzed, no change needed

An initial hypothesis (agents learn to never irrigate) was **refuted by
measurement**. Sweeping the true reward `crop_satisfaction − penalty·violation`
over the action, per regime:

| Regime | Optimal action | Reward |
|---|---|---|
| Wet (rain covers the crop) | don't irrigate | 1.00 |
| Drought, full reservoir | irrigate fully | 0.75 |
| Drought, low reservoir (quota binds) | request the minimum viable | 0.045 |

The reward is coherent. The real subtlety: the quota only bites when
`reservoir_level_norm < fixed_quota`, i.e. under reservoir drawdown — which only
happens with **real Raven dynamics**. In fallback mode the regulator has almost
no leverage, so the outer problem is nearly flat by construction. Studying /
tuning the reward therefore needs Raven, not a code change.

### 4. Inert ES dimensions — `mechanism.py`

`max_farm_area_m2` (a farm/world property, read from `ecology_cfg`, not from the
mechanism) and `under_irrigation_penalty_scale` (only referenced by commented-out
code) were in `optimize_params` but do not affect fitness. They are removed from
`optimize_params` (ES dimension 8 → 6) but kept in `ALL_PARAMS`/defaults, so the
mechanism vector and the observation are unchanged (still 12-dim obs).

### 5. Ray `local_mode` — `runtime.py`, `bilevel.py`

`RayRuntime` hard-coded `local_mode=True`. On Ray 2.53, running from inside the
editable-installed repo makes Ray auto-capture the repo as `working_dir`, which
local mode cannot upload → `ray.init` raises `"... is not a valid URI"`. It is now
a config field (default `False`) and a `BilevelConfig.ray(local_mode=...)` arg.
Real actors work under `uv` (Ray ships the package to workers). Set `True` only
for step-through debugging, and run from outside the repo if you do.

---

## Running the real experiment (`debug.py`)

`debug.py` now resolves Raven paths from the environment:

```bash
export USE_RAVEN=1
export RAVEN_CWD=/path/to/your/raven            # the ohms_canshield model directory
export RAVEN_CMD=/path/to/raven-binary          # e.g. a built Raven executable
WANDB_MODE=disabled PYTHONPATH=. uv run python examples/fresh_water/debug.py
```

If `RAVEN_CWD`/`RAVEN_CMD` are missing, the env logs a warning and runs in
fallback mode instead of crashing.

**What a real run needs (not in this repo):**
- The `ohms_canshield` Raven model directory (`.rvi/.rvt/.rvh`, `input/Extraction.rvt`, …).
- A built Raven binary. `raven-hydro` (PyPI) provides one but needs a CMake
  toolchain to build; it did not install out-of-the-box on macOS here.

---

## Porting these changes

All five commits are independent and localized. To adopt them elsewhere:

```bash
git cherry-pick 11c4518 34dda60 20c21f8 5f551c8 64d69ab
# or review the diff:
git diff master...fix/fresh-water-runnable -- \
  examples/fresh_water/regulated_env_ed_hs_v4.py \
  examples/fresh_water/regulator_env_raven.py \
  examples/fresh_water/mechanism.py \
  examples/fresh_water/debug.py \
  core/optimizers/es/optimizer.py \
  core/adaptors/ray/runtime.py \
  core/optimizers/bilevel.py
```

The ES fix (§2) and the Ray fix (§5) are the most broadly useful — they are not
fresh-water specific.

---

## Known remaining issues (not addressed here)

- **Registry vs. entry point mismatch.** `core/registry.py` maps
  `WaterRegulatedEdHsEnv` to `regulated_env_ed_hs.py` (the *original*), which still
  has a `NameError` in `violation_signal` (uses `underuse_penalty` /
  `stock_shortage_severity`, only defined in comments). `debug.py` sidesteps this by
  importing `regulated_env_ed_hs_v4.py` directly. The YAML/registry-driven path is
  therefore still broken — either point the registry at v4 or fix the original.
- **Raven toolchain / model files** — see above.
- Pre-existing lint (`I001`/`E402`/`F401`) in several files was left untouched.
