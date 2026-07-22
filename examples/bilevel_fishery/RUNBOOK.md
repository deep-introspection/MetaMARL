# Bilevel fishery — runnable + ES-convergence fixes

Branch: `fix/bilevel-fishery-debug` (derived from `feat/fresh-water-rework` @ `c8e2499`).

This branch takes `examples/bilevel_fishery/debug.py` from **crashes on a clean
checkout** to **runs end-to-end with an outer ES that actually optimizes**, and
fixes why running it **froze the whole laptop** (root cause: Ray's per-task
`setproctitle` hammering `launchservicesd` on macOS — see below).
Every change is small, localized, and justified so it can be cherry-picked into
the main line.

> Nothing here was committed to `feat/fresh-water-rework` (Nadine's branch) — all
> work lives on this parallel branch.

---

## TL;DR — what changed and why

| Commit | File(s) | Change | Why it mattered |
|---|---|---|---|
| `fix(ray): make local_mode/log_to_driver configurable` | `core/adaptors/ray/runtime.py`, `core/optimizers/bilevel.py` | `RayRuntimeConfig` gains `local_mode`/`log_to_driver` fields (default `False`); `.ray()` passes them through | `ray.init(local_mode=True)` was hard-coded. On Ray 2.53, launching from inside the editable repo makes Ray auto-capture the repo as `working_dir`, which local mode cannot upload → `ValueError: ... is not a valid URI`. Every run crashed at startup. |
| `fix(ray): cap BLAS/torch threads and propagate to workers` | `core/adaptors/ray/runtime.py` | Set `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR = omp_threads` on the driver **and** in every worker via `runtime_env["env_vars"]`; `torch.set_num_threads` on the driver | Each of ~30 Ray processes span one math-thread per core (numpy → Apple Accelerate ignores `OMP_NUM_THREADS`; torch defaulted to 12). ~1083 threads on 16 cores saturated the run-queue and starved the macOS UI. |
| `fix(es): let ES reporting support population > 1` | `core/optimizers/es/optimizer.py` | Feed the generation's **best** candidate to `plot_es_population` | `plot_es_population` validates `population.shape[0] == 1` (written for the pop=1 era). With a real population the ES passed the full array → `ValueError: population_size=16`, crashing the first outer iteration. |
| `feat(bilevel-fishery): set ES population to 16` | `examples/bilevel_fishery/debug.py` | `num_envs_per_env_runner = 16` | ES population `= num_envs_per_env_runner // len(inner seeds) = 1//1 = 1`. Fitness whitening over one sample gives a zero gradient → `mean = best = worst`, `var = 0`; the outer search never moved. |
| `fix(ray): stop uploading repo/.venv; disable RAY_DEBUG` | `core/adaptors/ray/runtime.py` | Inject repo onto worker `PYTHONPATH`; default `ray_debug=False` | Launching via `uv run` makes Ray upload the whole cwd (incl. the ~1.2 GB `.venv`) per run; `excludes` are ignored on this version → `/tmp/ray` grew to ~21 GB. Workers import from disk instead. |
| `fix(ray): disable per-task setproctitle on macOS` | `core/adaptors/ray/runtime.py`, `core/adaptors/ray/_worker_hooks.py` | No-op `ray._raylet.setproctitle` in every worker (`worker_process_setup_hook`) + driver | **The freeze.** Ray renames each worker per task; on macOS that's a synchronous XPC to `launchservicesd`, which saturates + holds the Launch Services lock → UI freezes (CPU idle). launchservicesd 190–214 % → ~0 %. |

**Not changed on purpose:** the reward, the inner APPO hyper-parameters, and the
scale (`outer_iters=1000`, `train_iters=100`). Those are methodological choices.

---

## Does it run?

Yes — launch with the **venv python directly, not `uv run`** (see the freeze
section below for why):

```bash
WANDB_MODE=disabled PYTHONPATH=. .venv/bin/python examples/bilevel_fishery/debug.py
```

With `reporter="wandb"` and `wandb login` done, drop `WANDB_MODE=disabled` for real
logging. `debug.py` does not call `logging.basicConfig`, so the driver-side
`[Bilevel]/[ES]/[PPO]` INFO logs are silent on stdout; wrap the launch in a
`logging.basicConfig(level=logging.INFO)` shim to watch progress, or follow wandb.

**Verified end-to-end** (a few outer iterations, reduced `train_iters` for speed):
inner APPO trains → `aggregate_rewards` → `[ES] gen=1 | best=0.456 | var=0.0049`,
`mean_norm 1.2247 → 1.2346`, `sigma 0.500 → 0.503` → `Outer iteration 2`. The
outer loop closes and the search center moves — Option A works.

---

## The changes in detail

### 1. Startup crash — Ray `local_mode` (`runtime.py`, `bilevel.py`)
`RayRuntimeConfig.initialize()` hard-coded `ray.init(local_mode=True)`. On Ray
2.53, running from inside the editable-installed repo makes Ray auto-capture the
repo as `working_dir`, which local mode cannot upload → `ray.init` raises
`"... is not a valid URI"`. `local_mode`/`log_to_driver` are now config fields
(default `False`) threaded through `BilevelConfig.ray()`. Set `local_mode=True`
only for step-debugging **and** launch from outside the repo.

### 2. ES population was 1 — degenerate outer loop (`debug.py`)
ES population is not a free knob: `bilevel.py` forces
`outer.batch_capacity = inner.batch_capacity`, and
`batch_capacity = num_envs_per_env_runner // len(inner seeds)`. `debug.py` had
`num_envs=1`, `num_seeds=1` → population 1. Whitening `(f - mean)/std` over one
sample ≈ 0 → zero ES gradient. Raised `num_envs_per_env_runner` to 16 (population
16, 8 antithetic pairs; must stay even while `break_symmetry=False`). Measured:
diverse fitness (`best_obj 0.456`, `worst_obj 0.203`, `var > 0`), mean/sigma now
move across generations.

### 3. ES reporting assumed population 1 (`es/optimizer.py`)
`core/reporting/utils/es_population.py::_validate_inputs` requires
`population.shape[0] == 1`. The ES now passes the full `[P, dim]` population →
crash. Quick unblock: pass only the best candidate per generation to
`plot_es_population`. **TODO:** extend the reporter to visualise all P candidates.

### 4. Machine freeze — see next section.

---

## Why running it froze the laptop (16-core Apple Silicon, 64 GB)

Three mechanisms — **all now fixed** (the third was the real killer):

- **Thread oversubscription (fixed).** ~30 Ray processes × ~12–16 math threads =
  ~1083 threads on 16 cores. numpy links Apple Accelerate, which **ignores
  `OMP_NUM_THREADS`** — only `VECLIB_MAXIMUM_THREADS` limits it — and torch
  defaulted to 12 intra-op threads. Nothing was capped effectively. Fixed by
  capping all BLAS/OMP vars + torch and propagating them to workers via
  `runtime_env["env_vars"]`. A/B microbench (8 procs): uncapped ~1 % idle CPU
  (frozen) → capped ~26 % idle (responsive).
- **1.2 GB working-dir upload per run (fixed).** Launching via `uv run` makes Ray
  auto-capture and upload the entire cwd — including the ~1.2 GB `.venv` (thousands
  of binaries) — into `/tmp/ray` on the first task. Ray's `excludes` are silently
  ignored here. Over the session `/tmp/ray` grew to **21 GB**. Fixed by launching
  with `.venv/bin/python` (no upload) and injecting the repo onto worker
  `PYTHONPATH` so they import from disk. Result: `/tmp/ray` 1.2 GB → **272 KB** per run.
- **`launchservicesd` flood via Ray's per-task `setproctitle` (fixed — this was
  the freeze).** Confirmed by sampling a live `ray::PolicyActor.train` worker:
  `execute_task → _changeproctitle → ray._raylet.setproctitle →
  darwin_set_process_title → _LSSetApplicationInformationItem →
  xpc_connection_send_message_with_reply_sync`. Ray renames each worker to
  `ray::<Task>` before every task (and back after); on macOS that vendored
  `setproctitle` does a **synchronous XPC round-trip to `launchservicesd` per
  call**. With ~19 workers doing it per task, launchservicesd's single serial
  queue saturates and holds the Launch Services lock → the whole UI freezes while
  the CPU stays 50–90 % idle. Fixed by no-op'ing the (cosmetic) rename in every
  worker via a `worker_process_setup_hook` + patching the driver. Measured:
  launchservicesd **190–214 % → ~0 %** across a full run; training proceeds
  normally. If a stale wedge lingers from an old run, clear it with
  `sudo killall launchservicesd` or a reboot.

The macOS **load average is unreliable here** — observed at 100–200 while the CPU
was 90 % idle and the machine empty. Judge health by **% CPU idle** and swap, not
load average.

---

## Running the real experiment

With the `setproctitle` fix, the run no longer freezes the laptop — it trains
with `launchservicesd` at ~0 % and the CPU with headroom. `debug.py` as committed
is still heavy (`outer_iters=1000`, `train_iters=100`, population 16, 5 agents;
one outer iteration ≈ several minutes with `num_env_runners=0`), so it is a long
run, but a runnable one on this machine.

```bash
# launch (VS Code Run and Debug works too — same venv interpreter, no uv):
WANDB_MODE=disabled PYTHONPATH=. .venv/bin/python examples/bilevel_fishery/debug.py
# stop it yourself:  Ctrl-C, or
pkill -9 -f 'bilevel_fishery/debug.py|ray::|default_worker|raylet'
```

Launch with the venv python directly (or VS Code debugpy), **not `uv run`** — `uv
run` makes Ray upload the repo/.venv into /tmp/ray.

---

## Known remaining issues (not addressed here)

- **Full 1000-iter run not run to completion** — it now runs without freezing
  (setproctitle fix), but it is long; only enough outer iterations were run to
  validate Option A functionally and confirm the freeze is gone.
- **Dead `num_seeds`.** The outer `.debugging(num_seeds=10)` is overridden by
  `bilevel.py:159-164` with the inner's single seed, so each mechanism is scored
  on one noisy seed. The "10 seeds" intent is lost.
- **`plot_es_population`** still only plots one candidate (the best) per generation.
- **`policy_loss=NA`** every inner iteration — the loss is not extracted from APPO
  results (instrumentation, benign).
- **ES reporter** is the only place population>1 was papered over; a proper
  population plot is a follow-up.
