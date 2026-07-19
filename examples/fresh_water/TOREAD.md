# Please read this first

This note explains, in plain language, everything that was changed in the
fresh-water part of this project, why it was changed, and what you should expect
now. It also lists honestly what could **not** be done. No prior knowledge of the
code is assumed.

A separate, more technical version for developers is in `RUNBOOK.md` (same folder).

---

## The short story

The fresh-water experiment has two connected parts working together:

- An **inner part**: many "water users" (think: farms) that learn how much water
  to request, using reinforcement learning.
- An **outer part**: a "regulator" that searches for the best set of rules
  (quotas, fines) to impose on those users. It searches by trial-and-error,
  keeping what works and adjusting from there.

Two things were broken:

1. **The experiment stopped immediately with an error** and could not run at all.
2. **The regulator's search never settled down** — it kept trying rules at full
   randomness forever, so it never homed in on a good answer. This is the
   "convergence problem" you mentioned.

Both are now fixed. The whole experiment runs from start to finish, and the
regulator's search now calms down and settles, as it should.

Below, each change is described on its own.

---

## Change 1 — The experiment no longer crashes at startup

**What was wrong.** The water users' environment relied on an external water
simulator (a program called **Raven**) to tell it the state of the river and the
reservoir at every step. The code assumed Raven's answers were **always**
available. The moment they were missing — for example on a computer where Raven
isn't installed, or when its output file wasn't produced — the program tried to
use information that didn't exist and stopped with an error. A comment in the code
claimed there was a "fallback" for this situation, but that fallback had never
actually been written.

**What we changed.** We made every piece of information have a sensible default
value *before* the code tries to get it from Raven. If Raven answers, its real
values are used. If Raven is missing or fails, the last known values are simply
kept (the river/reservoir is treated as unchanged), and a clear warning is
printed. The regulator's "how different is the river now?" measurement, which also
depended on Raven files, now returns a neutral value instead of crashing when
those files aren't there.

**Why.** So the experiment can be run and studied on any computer, even without
the Raven simulator installed.

**What to expect.** On a computer without Raven, the experiment runs to completion
and prints a warning that it is running in a simplified, **non-realistic** mode
(see the honest limitations at the end). On a computer that has Raven and the
model files, it uses Raven exactly as before.

## Change 2 — The regulator's search now settles down (the convergence fix)

**What was wrong.** The regulator explores rules by making random variations
around its current best guess, and shrinking how random it is as it becomes more
confident. Two mistakes stopped it from ever becoming confident:

- It judged "am I succeeding?" using numbers that had been mathematically
  re-centred so that, by construction, roughly half of them always looked like
  "successes" — no matter how well or badly it was actually doing. So its
  confidence signal was meaningless.
- Separately, a line of code kept dragging the amount of randomness back up toward
  a middle value every single round, so it could never shrink.

Together, these meant the search stayed maximally random forever and never
narrowed in on a good set of rules.

**What we changed.** The success signal is now measured on the real scores,
compared against the best score seen so far — so as the regulator gets close to a
good answer, it correctly sees that new attempts rarely beat it, and it calms
down. We removed the line that kept forcing the randomness back up.

**Why.** So the outer search actually converges instead of wandering forever.

**What to expect.** We tested this in isolation on a simple problem with a known
best answer. Before the fix, the randomness stayed stuck near its maximum and the
best guess kept jittering around without settling. After the fix, the randomness
steadily shrank (from 0.5 down to about 0.05) and the best guess homed in on the
correct answer. This is the core fix for the convergence problem.

## Change 3 — The regulator no longer wastes effort on knobs that do nothing

**What was wrong.** The regulator was searching over eight "knobs". Two of them
had no effect on the outcome whatsoever: one was the size of a farm (which is a
property of the world, not something the regulator sets, and the code read it from
a fixed setting anyway), and the other was only mentioned in commented-out
(disabled) code. Searching over knobs that change nothing wastes effort and makes
the search harder.

**What we changed.** The regulator now searches over the six knobs that actually
matter. The two inert ones are kept at fixed default values, so nothing else about
the experiment changes.

**Why.** A smaller, meaningful search space is easier to optimise.

**What to expect.** Same behaviour, but the search is cleaner and a little easier.

## Change 4 — The experiment is no longer tied to one person's computer

**What was wrong.** The location of the Raven simulator was written directly into
the code as a fixed path on one specific person's machine (including a Windows
`.exe` file). On any other computer this path is wrong.

**What we changed.** The locations are now read from environment settings
(`RAVEN_CWD`, `RAVEN_CMD`, and `USE_RAVEN` to turn Raven on/off), with a sensible
default. If the simulator or its files can't be found, the experiment prints a
clear message and automatically switches to the simplified mode instead of
failing.

**Why.** So anyone can run it on their own machine, and so it fails gracefully
with a clear explanation rather than a confusing error.

**What to expect.** Point those settings at your own Raven installation to use it;
leave them unset to run in simplified mode.

## Change 5 — Fixing the crash when starting the computation engine

**What was wrong.** The project uses an engine called **Ray** to run many things
at once. The code forced Ray into a special single-process "debugging" mode. In
the current version of Ray, that mode crashes at startup when the program is
launched from inside the project folder. So the experiment couldn't even start.

**What we changed.** That debugging mode is now off by default (and can still be
switched on when someone specifically needs to step through the code).

**Why.** So the experiment starts normally.

**What to expect.** It starts and runs. We confirmed the entire experiment runs
from beginning to end this way.

---

## About the reward (what we deliberately did NOT change)

You asked us to also look at how users are "rewarded" for their water decisions,
because it looked like it might push them to never irrigate. **We checked this
carefully by measuring the actual reward across situations, and it turned out to
be sound**, so we changed nothing:

- When it rains enough, the crop is satisfied and the best choice is to not
  irrigate — which is correct.
- In a drought with a full reservoir, the best choice is to irrigate fully —
  correct.
- In a drought with a low reservoir, the rules correctly limit how much can be
  taken.

Our earlier worry was a guess; the measurement disproved it. The one real
subtlety: the regulator's quota only "bites" when the reservoir is drawn down, and
that only happens with the real Raven simulator. So the reward can only be studied
properly **with Raven running** — it is not something to fix in the code.

---

## How to run it

**Simplified mode (no simulator needed), a quick check that everything works:**

```
WANDB_MODE=disabled PYTHONPATH=. uv run python examples/fresh_water/smoke_run.py
```

It should end with the line `CHAIN OK`. This runs a small version of the whole
experiment (a few users, two rounds) just to confirm nothing is broken.

**The full experiment:**

```
WANDB_MODE=disabled PYTHONPATH=. uv run python examples/fresh_water/debug.py
```

This is heavy (it uses 500 users) and, without Raven, runs in the simplified,
non-realistic mode.

**To use the real Raven simulator**, set these before running:

```
export USE_RAVEN=1
export RAVEN_CWD=/path/to/your/raven/model/folder
export RAVEN_CMD=/path/to/your/raven/program
```

---

## What we could NOT do (honest limitations)

- **We could not run a scientifically meaningful experiment**, because that needs
  the Raven water-model files (the "ohms_canshield" model) and a working Raven
  program. **Neither is included in this project** — the model files live only on
  the original author's computer, and the Raven program did not install on our
  machine (it needs to be compiled with extra build tools). So everything we ran
  here was in the **simplified mode**, which is only good for checking that the
  code works, not for real results.

- **The simplified mode is not physically realistic.** In it, the river and
  reservoir simply stay still, so the numbers coming out are meaningless as
  science. It exists only to let the program run end-to-end. Every run in this mode
  prints a warning saying exactly this.

- **We did not change how the users are rewarded** (see the reward section above),
  because measurement showed it was already sound. Changing it would have been an
  unjustified change to the scientific method.

- **We did not remove the large amount of unused/old code** we found (see next
  section) — that deletion is waiting for your go-ahead, since it is your team's
  code and some of it may be intentional history.

---

## Unused / leftover code we found (cleanup proposed, not yet done)

While working, we found several files in this folder that nothing in the running
experiment uses. We have **not** deleted anything yet. Here is the inventory:

- **Old versions of the users' environment**, superseded by the current one
  (`regulated_env_ed_hs_v4.py`): `regulated_env_ed_hs-v2.py`,
  `regulated_env_ed_hs_v3.py`, `regulated_env_ed_hs_v4_no_quota.py`, and
  `regulated_env_raven.py`. Nothing imports these.

- **An alternative "load settings from a YAML file" starter** (`bilevel.py`) plus
  the two YAML files it reads (`config_ed_hs.yaml`, `config_run_raven.yaml`).
  Nothing uses this starter, and it also contains a bug (it looks for "fisher"
  users while the water configs define "utilizer" users), so it would fail if used.

- **A naming trap.** The project's internal directory (`core/registry.py`) still
  points the name `WaterRegulatedEdHsEnv` at the *old, broken* file
  (`regulated_env_ed_hs.py`), even though the running experiment uses the newer
  `..._v4.py`. The old file has a genuine bug (it refers to values that were left
  commented out). Anyone using the name-based lookup would hit that bug. The clean
  fix is to point the directory at the current file, or remove the old one.

- The `deprecated/` sub-folder is already labelled as old and can be left as-is or
  removed.

We can remove any or all of these on your say-so.
