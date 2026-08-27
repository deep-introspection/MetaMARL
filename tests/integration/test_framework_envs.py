"""Integration tests for the optimizer <-> environment <-> World plumbing.

These tests use the real ``World`` Ray actor and a ``FakeReporter`` actor in
place of Weights & Biases. They check that an outer ES optimizer built from
``ESConfig`` drives a ``RegulatorEnv`` through the World and converges on an
analytic fitness landscape.
"""

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import ray

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.optimizers.es.config import ESConfig
from core.world.base import World
from tests.conftest import FakeReporter


@pytest.mark.integration
def test_es_regulator_loop(ray_session):
    world = World.options(name=f"world_{uuid.uuid4().hex[:8]}").remote()
    reporter = FakeReporter.remote()

    optimum = np.array([0.8, 0.2, 0.6], dtype=np.float32)

    class AnalyticRegulatorEnv(RegulatorEnv):
        """Regulator env with no inner optimizer: fitness is a closed form."""

        def __init__(self, *, optimum: np.ndarray, optimizer=None, **kwargs):
            # the env creator always injects ``optimizer``; the analytic path has none
            super().__init__(optimizer=None, **kwargs)
            self.optimum = optimum

        @override(RegulatorEnv)
        def _pre_reset(self, seed=None):
            pass

        @override(RegulatorEnv)
        def _step(self, theta: np.ndarray):
            x = np.asarray(theta, dtype=np.float32)
            fitness = -np.sum((x - self.optimum) ** 2, axis=1)
            return None, fitness, False, False, {}

        @override(RegulatorEnv)
        def reward(self, reward: np.ndarray) -> np.ndarray:
            return reward

        @override(RegulatorEnv)
        def aggregate_rewards(self, ctxs):
            raise AssertionError("not used on the analytic path")

    es_cfg: ESConfig = (
        ESConfig()
        .training(sigma=0.2, mean_lr=0.2, sigma_lr=0.0, break_symmetry=True)
        .environment(
            env=AnalyticRegulatorEnv,
            env_config={
                "optimum": optimum,
                # ES reads the optimized parameter names from the env's space
                "mechanism_space": SimpleNamespace(
                    optimize_params=[f"p{i}" for i in range(optimum.shape[0])]
                ),
            },
        )
        .debugging(seed=0)
    )
    es_cfg.dimension = optimum.shape[0]

    es = es_cfg.build_optimizer(world=world, reporting=reporter)
    es.batch_capacity = 16

    initial_dist = np.linalg.norm(es.mean - optimum)
    for _ in range(30):
        es.run()
    final_dist = np.linalg.norm(es.mean - optimum)

    assert es.generation == 30
    assert final_dist < 0.5 * initial_dist
    # every generation publishes one EnvStepContext for this optimizer
    assert len(ray.get(world.get_opt_ctx_ids.remote(es.id))) == 30
    assert ray.get(reporter.get_calls.remote())["plot_es_population"] == 30


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Written against the pre-`core` mechanism-space API; the RLlib env "
        "creator now requires a BilevelConfig-managed mechanism. Replaced by "
        "per-feature integration tests (tests/integration/test_fishery_*.py)."
    )
)
def test_ppo_with_regulated_env():
    pass


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Written against the pre-`core` mechanism-space API; superseded by the "
        "per-feature bilevel smoke tests."
    )
)
def test_full_bilevel_es_ppo_loop():
    pass
