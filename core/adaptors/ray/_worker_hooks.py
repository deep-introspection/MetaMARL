"""Ray worker setup hooks.

Wired into Ray via ``runtime_env["worker_process_setup_hook"]`` so it runs once,
at worker start-up, before any task executes.
"""


def silence_setproctitle() -> None:
    """Make Ray's per-task ``setproctitle`` a no-op.

    Ray renames every worker to ``ray::<Task>`` before running each task and back
    to ``ray::IDLE`` after (see ``ray._private.worker._changeproctitle``). On
    macOS the vendored ``setproctitle`` -> ``darwin_set_process_title`` ->
    ``_LSSetApplicationInformationItem`` performs a *synchronous* XPC round-trip
    to ``launchservicesd`` on every call. With several workers each doing this
    per task, launchservicesd's single serial queue saturates and holds the
    Launch Services lock, so the whole macOS UI freezes even while the CPU is
    idle (confirmed by sampling a ``ray::PolicyActor.train`` worker).

    The rename is purely cosmetic (Activity Monitor labels), so we drop it.
    Reassigning the module attribute works because Ray looks it up as
    ``ray._raylet.setproctitle(...)`` at call time.
    """
    try:
        import ray

        ray._raylet.setproctitle = lambda *args, **kwargs: None
    except Exception:
        # Never let a cosmetic tweak break worker start-up.
        pass
