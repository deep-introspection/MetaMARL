"""``RayRuntimeConfig``: environment variables and the arguments handed to ``ray.init``."""

import os

import pytest
import ray
from ray._private import ray_constants

from core.adaptors.ray.runtime import RayRuntime, RayRuntimeConfig


@pytest.fixture
def captured_init(monkeypatch):
    calls = []
    monkeypatch.setattr(ray, "init", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    return calls


@pytest.mark.unit
def test_initialize_passes_resources_and_disables_uv_hook(captured_init, monkeypatch):
    monkeypatch.setattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", True)
    cfg = RayRuntimeConfig(
        device="cpu",
        num_cpus=3,
        runtime_env={"excludes": ["x"]},
        init_kwargs={"foo": 1},
    )
    cfg.initialize()

    assert len(captured_init) == 1
    kwargs = captured_init[0]
    assert kwargs["num_cpus"] == 3 and "num_gpus" not in kwargs
    assert kwargs["local_mode"] is True and kwargs["foo"] == 1
    assert kwargs["runtime_env"] == {"excludes": ["x"]}
    assert ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV is False
    assert (
        os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["RLLIB_NUM_GPUS"] == "0"
    )


@pytest.mark.unit
def test_ensure_initialized_is_idempotent(captured_init, monkeypatch):
    RayRuntime.ensure_initialized(RayRuntimeConfig())
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    RayRuntime.ensure_initialized(RayRuntimeConfig())
    assert len(captured_init) == 1
