# Please read this first

This note explains, in plain language, what we changed in the **bilevel fishery**
experiment (`examples/bilevel_fishery/debug.py`), why, and what to expect now. It
also says honestly what we could **not** finish. No prior knowledge of the code is
assumed.

A more technical version for developers is in `RUNBOOK.md` (same folder).

**Important:** none of this was put on your branch (`feat/fresh-water-rework`). All
of it lives on a separate branch, `fix/bilevel-fishery-debug`, so your work is
untouched. You can look at it whenever you want and keep what you like.

---

## The short story

The bilevel fishery experiment has two parts working together:

- An **inner part**: many "fishers" that learn how much to fish, by trial and error.
- An **outer part**: a "regulator" that searches for the best rules (quotas, fines)
  to impose on them, keeping what works and adjusting from there.

Three things were wrong. All three are now fixed:

1. **The experiment crashed the moment it started** and could not run at all.
2. **The regulator's search never really searched** — it kept proposing essentially
   the same single rule every round, so it could never learn what works.
3. **Running it froze the computer**, to the point of being unusable.

The experiment now runs on a small scale, and the regulator's search genuinely
explores and improves. The freezing is now fully understood **and fixed**: it came
from the Ray library renaming its workers on macOS in a way that overwhelmed a
core system service (details in Change 4). With that switched off, the machine
stays responsive during a run.

---

## Change 1 — It no longer crashes at startup

**What was wrong.** The engine that runs everything in parallel (called **Ray**)
was forced into a special single-process "debug" mode. In the current version of
Ray, that mode crashes at startup when the program is launched from inside the
project folder.

**What we changed.** That mode is now off by default (and can still be switched on
when someone specifically needs to step through the code).

**What to expect.** It starts and runs normally.

## Change 2 — The regulator now actually searches (not just one rule)

**What was wrong.** The regulator is supposed to try a *batch* of candidate rules
each round, compare them, and move toward the good ones. Because of a settings
mismatch, it was only ever trying **one** rule per round. With a single candidate
there is nothing to compare, so its "which direction is better?" signal was always
zero — the search stood still.

**What we changed.** We set it to try **16** candidate rules per round.

**What to expect.** We checked this and confirmed it: the 16 candidates now get
genuinely different scores (the best clearly better than the worst), the search
signal is no longer zero, and the regulator's current best guess visibly moves from
one round to the next. This is the core fix that makes the outer search meaningful.

## Change 3 — Running it no longer overloads the machine

**What was wrong.** Two separate things were overloading the computer:

- **Too many threads.** Every parallel worker was trying to use *all* the
  computer's cores at once for its maths. With ~30 workers, that meant over a
  thousand threads fighting for 16 cores — so the whole interface froze even though
  no single program looked busy. (A quirk of macOS made this worse: the maths
  library Apple ships ignores the usual "use one core" setting.)
- **A 1.2 GB copy on every run.** Because of the way the program was launched, Ray
  was silently copying the *entire* project — including the 1.2 GB of installed
  software — into a temporary folder **every single time it ran**. Over the day that
  temporary folder ballooned to 21 GB.

**What we changed.** We capped each worker to a single core (with the correct macOS
setting), and we stopped the useless 1.2 GB copy — the workers now read the code
directly from disk. The temporary folder now stays at a few hundred kilobytes
instead of gigabytes.

**What to expect.** During a run the processor now keeps plenty of spare capacity
instead of being pinned at 100 %, and the disk no longer fills up.

## Change 4 — The real cause of the freezing (now fixed)

This was the big one, and it took the longest to find.

**What was wrong.** So that each parallel worker shows a helpful name in Activity
Monitor (like `ray::PolicyActor.train`), the Ray library **renames every worker
before and after each small task it runs**. On macOS that rename is *not* free:
each one sends a request to a core system service — `launchservicesd`, the one
that keeps track of applications — and waits for a reply. With ~19 workers
renaming themselves thousands of times during training, that single system
service gets swamped. And because other parts of macOS — including the on-screen
interface — have to queue behind the same service, **the whole machine freezes,
even though the processor itself is mostly idle**. That's why the processor always
looked free, why closing other apps didn't help, and why it happened on a freshly
rebooted machine too.

**How we found it.** We captured a live snapshot of one worker *while the machine
was frozen*; its internal call trace pointed straight from "rename this worker" to
that system service.

**What we changed.** The rename is purely cosmetic (just the label in Activity
Monitor), so we switched it off in every worker.

**What to expect.** That system service now stays near **0 %** during a run (it was
pinned at 200–300 %), and the machine stays responsive. Measured before/after on a
full run — this is the fix for the freezing.

(If that service is ever still stuck from an *earlier* run, `sudo killall
launchservicesd` or a reboot clears it — but a fresh run no longer causes it.)

---

## How to run it

Launch it like this — note it is **not** `uv run` (that is what caused the 1.2 GB
copy):

```
WANDB_MODE=disabled PYTHONPATH=. .venv/bin/python examples/bilevel_fishery/debug.py
```

Remove `WANDB_MODE=disabled` (and run `wandb login` once) to log to Weights &
Biases as before.

To stop it yourself at any time: press **Ctrl-C** in its terminal, or run
`pkill -9 -f 'bilevel_fishery/debug.py|ray::(World|PolicyActor|MultiAgentEnvRunner|WandbReporter)'`.

---

## What we could NOT do (honest limitations)

- **We did not run the full experiment to the very end.** It now runs on the
  laptop **without freezing**, but it is a long run (1000 outer rounds × 100 inner
  steps each). We ran enough rounds to confirm the mechanics work and that the
  freeze is gone — not to produce final scientific results. It can now be left to
  run (locally or elsewhere) as you see fit.

- **We did not change the reward, the fishers' learning settings, or the scale.**
  Those are your scientific choices; we only fixed what was broken.

- **A few smaller quirks were left as-is** (see `RUNBOOK.md`): the "10 seeds"
  setting on the outer loop is silently ignored (each rule is scored on a single
  seed), the population plot only shows the best candidate per round, and one
  training metric (`policy_loss`) is not being reported. None of these block the run.
