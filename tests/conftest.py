"""Shared pytest fixtures.

The framework exchanges state through a Ray actor (``core.world.base.World``).
Unit tests replace it with ``FakeWorld``, an in-memory stand-in exposing the
same ``<method>.remote(...)`` call shape, and make ``ray.get`` a pass-through so
that no Ray runtime is needed. Integration tests use the real actor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import ray


class FakeWorld:
    """In-memory replacement for the ``World`` Ray actor.

    Records every published context in ``contexts`` and mimics the remote
    call interface (``world.append_context.remote(ctx)``) used by the
    environments, so that ``BaseEnv`` subclasses can be exercised without Ray.
    """

    def __init__(self) -> None:
        self.contexts: list = []
        self.flushed_ids: list = []
        self.flushed_status: list = []
        self.append_context = SimpleNamespace(remote=self._append_context)
        self.get_ctx_registry = SimpleNamespace(remote=self._get_ctx_registry)
        self.flush_ctx = SimpleNamespace(remote=self._flush_ctx)
        self.flush = SimpleNamespace(remote=self._flush)
        self.register_optimizer = SimpleNamespace(remote=self._register_optimizer)
        self.get_opt_registry = SimpleNamespace(remote=lambda: set(self.opt_ids))
        self._set_new_opt_id = SimpleNamespace(remote=self._set_new_opt_id_impl)
        self.opt_ids: list[str] = []

    def _append_context(self, ctx) -> None:
        self.contexts.append(ctx)

    def _get_ctx_registry(self) -> dict:
        return {i: ctx for i, ctx in enumerate(self.contexts)}

    def _flush_ctx(self, keys) -> None:
        self.flushed_ids.extend(list(keys))

    def _flush(self, status=None) -> None:
        self.flushed_status.append(status)

    def _register_optimizer(self, opt) -> str:
        return self._set_new_opt_id_impl(opt_id=getattr(opt, "opt_id", None))

    def _set_new_opt_id_impl(self, opt_id=None) -> str:
        opt_id = opt_id or f"opt_{len(self.opt_ids)}"
        self.opt_ids.append(opt_id)
        return opt_id


@pytest.fixture
def fake_world(monkeypatch) -> FakeWorld:
    """A ``FakeWorld`` with ``ray.get`` patched to return its argument."""
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    return FakeWorld()


@pytest.fixture(scope="session")
def ray_session():
    """Start a small local Ray runtime for integration tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


@ray.remote
class FakeReporter:
    """Ray actor standing in for ``WandbReporter``.

    Records how many times each plotting entry point was called instead of
    logging to Weights & Biases; the optimizers only require the call shape.
    """

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def _record(self, name: str) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1

    def plot_es_population(self, **kwargs) -> None:
        self._record("plot_es_population")

    def plot_ray_result(self, *args, **kwargs) -> None:
        self._record("plot_ray_result")

    def plot_env_step(self, *args, **kwargs) -> None:
        self._record("plot_env_step")

    def plot_env_reduced(self, *args, **kwargs) -> None:
        self._record("plot_env_reduced")

    def get_calls(self) -> dict[str, int]:
        return dict(self.calls)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run the tests that start a real Ray runtime after every other test.

    When a Ray actor class is exported to a live cluster it is serialized by
    value, and unpickling it back in the driver process makes cloudpickle
    rewrite the methods of the already-imported class with copies whose
    ``__globals__`` is a frozen snapshot of the module namespace. Any later
    ``monkeypatch.setattr(module, ...)`` in a unit test is then invisible to
    those methods (observed on ``core.reporting.wandb.WandbReporter``). Keeping
    the ``integration`` and ``notebook`` items last removes the order
    dependency without touching the production classes.
    """
    ray_items = [
        item
        for item in items
        if item.get_closest_marker("integration") or item.get_closest_marker("notebook")
    ]
    others = [item for item in items if item not in ray_items]
    items[:] = others + ray_items
