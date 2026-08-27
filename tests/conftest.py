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
        return self._set_new_opt_id_impl(opt_id=f"opt_{len(self.opt_ids)}")

    def _set_new_opt_id_impl(self, opt_id: str) -> str:
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
