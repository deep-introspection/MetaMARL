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

    def _append_context(self, ctx) -> None:
        self.contexts.append(ctx)

    def _get_ctx_registry(self) -> dict:
        return {i: ctx for i, ctx in enumerate(self.contexts)}

    def _flush_ctx(self, keys) -> None:
        self.flushed_ids.extend(list(keys))

    def _flush(self, status=None) -> None:
        self.flushed_status.append(status)


@pytest.fixture
def fake_world(monkeypatch) -> FakeWorld:
    """A ``FakeWorld`` with ``ray.get`` patched to return its argument."""
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)
    return FakeWorld()
