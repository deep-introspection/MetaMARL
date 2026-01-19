from typing import override

from core.optimizers.base import Optimizer
from core.optimizers.config import BaseOptimizerConfig

from gymnasium import spaces
import gym

import numpy as np

from core.world.base import World, Context
from core.world.context import ContextSchema


class SignalContext(ContextSchema):
    value: float


class DummyOptimizer(Optimizer):
    def run(self, world: World) -> None:
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        obs, reward, *_ = self.env.step(action)
        self.last_reward = reward


class SuperDummyOptimizer(Optimizer):
    def run(self, world: World) -> None:
        ctx = Context(
            id=None,
            opt_id=self.id,
            schema=SignalContext,
            payload=SignalContext(value=1.0),
        )
        world.set_new_context(ctx)


class DummyContextWrapper(ContextWrapper):
    def _get_violation_signal(self) -> float:
        ctx_ids = self._get_contexts()
        if not ctx_ids:
            return 0.0
        ctx = self._world._contexts[next(iter(ctx_ids))]
        return ctx.payload.value

    def _get_violation_penalty(self) -> float:
        return 1.0

    def observation(self, observation):
        return observation

    def action(self, action):
        return action


class DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32
        )
        self.state = np.zeros(2, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(2, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.state = action * 2.0
        reward = -float(np.linalg.norm(action))
        return self.state, reward, False, False, {}


class DummyOptimizerConfig(BaseOptimizerConfig):
    pass


class SuperDummyOptimizerConfig(BaseOptimizerConfig):
    pass


import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.wrappers.context_wrapper import ContextWrapper


def test_core_framework_end_to_end():
    # Setup Shared world
    world = World()

    # Setup Parent Optimizer
    parent_cfg = SuperDummyOptimizerConfig().world(world=world)
    parent = parent_cfg.build_optimizer()
    parent.set_id("parent")

    # Setup Child Optimizer
    child_env = DummyContextWrapper(env=DummyEnv(), world=world, opt_id=parent.id)
    child_cfg = DummyOptimizerConfig().world(world=world).environment(child_env)
    child = child_cfg.build_optimizer()
    child.set_id("child")

    # Manual Orchestration
    parent.run(world)
    child.run(world)
