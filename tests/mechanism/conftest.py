"""Fixtures shared by the mechanism tests."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def env_at():
    """Factory: a minimal env-like object exposing ``S_t["fish"]`` and ``K``."""

    def _make(fish_norm: float, K: float = 1000.0, **extra):
        return SimpleNamespace(S_t={"fish": fish_norm * K}, K=K, **extra)

    return _make


@pytest.fixture
def resource_binding():
    return {"resource_level": lambda env: env.S_t["fish"] / env.K}
