"""Shared fixtures for the reporting unit tests.

The reporting utilities under ``core.reporting`` never need a live Weights &
Biases run: ``wandb.Table``, ``wandb.Plotly`` and ``wandb.plot.line_series``
are plain data objects. The only run-side surface the code touches is
``run.log(...)`` and ``run.define_metric(...)``, so ``FakeRun`` records those
calls in memory. Every plotting module also keeps module-level table caches
keyed on ``id(run)``; the autouse ``_reset_reporting_caches`` fixture empties
them between tests so that each test starts from a clean history.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import core.reporting.utils.env_reduced as env_reduced_mod
import core.reporting.utils.env_step_context as env_step_mod
import core.reporting.utils.es_population as es_mod
import core.reporting.utils.ray_new_api_stack as ray_mod
from core.world.context import Context, EnvStepContext, MechanismStatus


class FakeRun:
    """In-memory stand-in for a ``wandb.Run``.

    Attributes
    ----------
    logs : list of tuple
        One ``(payload, step, commit)`` entry per ``log`` call.
    defined : list of tuple
        One ``(name, kwargs)`` entry per ``define_metric`` call.
    """

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, Any], int | None, bool | None]] = []
        self.defined: list[tuple[str, dict[str, Any]]] = []
        self.finished = 0

    def log(
        self,
        payload: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        self.logs.append((dict(payload), step, commit))

    def define_metric(self, name: str, **kwargs: Any) -> None:
        self.defined.append((name, dict(kwargs)))

    def finish(self) -> None:
        self.finished += 1

    # Helpers -------------------------------------------------------------
    def logged_keys(self) -> set[str]:
        keys: set[str] = set()
        for payload, _, _ in self.logs:
            keys.update(payload.keys())
        return keys

    def payload_for(self, key: str) -> Any:
        for payload, _, _ in self.logs:
            if key in payload:
                return payload[key]
        raise KeyError(key)


_CACHES = (
    env_step_mod._ENV_REWARD_TABLES,
    env_step_mod._ENV_ACTION_TABLES,
    env_step_mod._ENV_OBS_TABLES,
    env_step_mod._ENV_INFO_TABLES,
    env_reduced_mod._ENV_REDUCED_ITER_TABLES,
    es_mod._ES_HISTORY_TABLES,
    ray_mod._RETURNS_TABLES,
    ray_mod._TRAIN_EVAL_RETURN_TABLES,
    ray_mod._LEARNER_METRIC_TABLES,
)


@pytest.fixture(autouse=True)
def _reset_reporting_caches():
    """Empty the module-level ``id(run)``-keyed table caches around each test."""
    for cache in _CACHES:
        cache.clear()
    yield
    for cache in _CACHES:
        cache.clear()


@pytest.fixture
def fake_run() -> FakeRun:
    return FakeRun()


def make_env_ctx(
    *,
    step: int = 0,
    status: MechanismStatus = MechanismStatus.train,
    mechanism: int | None = 0,
    seed: int | None = 1,
    env_id: int | None = 0,
    observation: Any = None,
    observation_map: list[str] | None = None,
    reward: Any = None,
    action: Any = None,
    info: Any = None,
) -> Context:
    """Build a ``Context`` wrapping an ``EnvStepContext`` for one agent step.

    Defaults produce a two-agent fishery-like transition whose ``info`` keys
    all belong to the ``KEEP_METRICS`` allowlist of ``env_reduced``.
    """
    if observation is None:
        observation = {
            "fisher:0": np.array([10.0, 0.5]),
            "fisher:1": np.array([10.0, 0.6]),
        }
    if reward is None:
        reward = {"fisher:0": 1.0, "fisher:1": 2.0}
    if action is None:
        action = {"fisher:0": np.array([0.3]), "fisher:1": np.array([0.4])}
    if info is None:
        info = {
            "fisher:0": {"fish": 10.0, "H_realized": 0.3, "farm_area_m2": 100.0},
            "fisher:1": {"fish": 10.0, "H_realized": 0.4, "farm_area_m2": 300.0},
        }
    payload = EnvStepContext(
        env_id=env_id,
        seed=seed,
        policy_seed=seed,
        status=status,
        mechanism=mechanism,
        observation=observation,
        observation_map=observation_map,
        reward=reward,
        action=action,
        info=info,
    )
    return Context(
        id=f"ctx_{step}", opt_id="opt_0", step=step, env="fishery", payload=payload
    )


@pytest.fixture
def env_ctx() -> Context:
    return make_env_ctx(step=3)
