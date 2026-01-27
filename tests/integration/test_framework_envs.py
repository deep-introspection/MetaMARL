import numpy as np
import pytest
from gymnasium import spaces

from core.annotations import override
from core.envs.regulated import RegulatedEnv
from core.envs.regulator import RegulatorEnv
from core.world.base import World
from src.es.config import ESConfig
from src.ppo.config import PPOptimizerConfig


@pytest.mark.integration
def test_es_regulator_loop():
    dummy_world = World()
    optimum = np.array([0.8, 0.2, 0.6], dtype=np.float32)

    class DummyRegulatorEnv(RegulatorEnv):
        def __init__(self, *, world: World, optimum: np.ndarray, **kwargs):
            super().__init__(world=world, optimizer=None)
            self.optimum = optimum

        @override(RegulatorEnv)
        def _step(self, theta: np.ndarray):
            x = np.asarray(theta)
            fitness = -np.sum((x - self.optimum) ** 2, axis=1)
            return None, fitness, False, False, {}

        @override(RegulatorEnv)
        def reward(self, reward: np.ndarray) -> np.ndarray:
            return reward

    es_cfg: ESConfig = (
        ESConfig()
        .training(
            dimension=optimum.shape[0],
            pop_size=4,
            sigma=0.1,
            mean_lr=0.2,
            sigma_lr=0.0,
        )
        .environment(env=DummyRegulatorEnv, env_config={"optimum": optimum})
    )

    es = es_cfg.build_optimizer(world=dummy_world)

    # TODO we need a main orchestrator
    for _ in range(3):
        es.run()

    assert len(dummy_world.get_opt_ctx_ids(es.id)) > 0


@pytest.mark.integration
def test_ppo_with_regulated_env():
    world = World()

    class DummyRegulatedEnv(RegulatedEnv):
        def __init__(self, *, world, **kwargs):
            super().__init__(world=world)
            self.observation_space = spaces.Box(-1, 1, (4,), np.float32)
            self.action_space = spaces.Discrete(2)

        def _step(self, action):
            obs = np.random.randn(4).astype(np.float32)
            reward = 1.0
            return obs, reward, False, False, {}

        def _reset(self):
            return np.random.randn(4).astype(np.float32)

        def violation_signal(self):
            return 0.5

        def violation_penalty(self):
            return 0.2

    # TODO perhaps a world config fc
    cfg = (
        PPOptimizerConfig()
        .environment(env=DummyRegulatedEnv)
        .framework(framework="torch")
        .resources(num_gpus=0)
        .training(train_batch_size=200)
    )

    ppo = cfg.build_optimizer(world=world)

    # Run a few iterations
    for _ in range(3):
        result = ppo.run()

    assert "episode_reward_mean" in result
    assert len(world.get_opt_ctx_ids(ppo.id)) > 0


@pytest.mark.integration
def test_full_bilevel_es_ppo_loop():
    world = World()

    # --- Regulated Environment ---
    class DummyBilevelRegulatedEnv(RegulatedEnv):
        def __init__(self, world):
            super().__init__(world)
            self.observation_space = spaces.Box(-1, 1, (1,), np.float32)
            self.action_space = spaces.Discrete(2)

        def _step(self, action):
            ctx = next(iter(self.world._contexts.values()))
            theta = ctx.payload.theta.to_vector()[0]

            obs = np.array([theta], dtype=np.float32)
            reward = 1.0
            return obs, reward, False, False, {}

        def violation_signal(self):
            ctx = next(iter(self.world._contexts.values()))
            return abs(ctx.payload.theta.to_vector()[0])

        def violation_penalty(self):
            return 0.1

    ppo_cfg = (
        PPOptimizerConfig()
        .environment(env=DummyBilevelRegulatedEnv())
        .framework(framework="torch")
        .training(train_batch_size=200)
    )

    ppo = ppo_cfg.build_optimizer(world)

    # --- Regulator Environment ---
    # TODO passing a callable to the reward !
    class DummyBilevelRegulatorEnv(RegulatorEnv):
        @override(RegulatorEnv)
        def aggregate_rewards(self, rewards):
            return float(np.mean(rewards))

    es_cfg: ESConfig = (
        ESConfig()
        .training(
            dimension=1,
            pop_size=4,
            sigma=0.1,
            mean_lr=0.2,
            sigma_lr=0.0,
        )
        .environment(
            env=DummyBilevelRegulatorEnv(
                world=world, optimizer=ppo, train_iters=2, eval_iters=2
            )
        )
    )

    es = es_cfg.build_optimizer()

    # --- Run full bilevel loop ---
    for _ in range(5):
        es.run()

    assert len(world.get_ctx_ids()) > 0


test_ppo_with_regulated_env()
