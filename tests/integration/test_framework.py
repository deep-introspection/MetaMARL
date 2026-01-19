import numpy as np
import pytest
from gymnasium import Env, spaces

from core.optimizers.base import Optimizer
from core.optimizers.config import BaseOptimizerConfig
from core.world.base import Context, World
from core.world.context import ContextSchema
from core.wrappers.context_wrapper import ContextWrapper


class SignalContext(ContextSchema):
    value: float

# TODO 
class SuperDummyOptimizer(Optimizer):
    def run(self, world: World) -> None:
        ctx = Context(
            id="regulation_signal", 
            opt_id=self.id, 
            payload=SignalContext(value=1.0)
        )
        if ctx.id not in world.get_opt_ctx_ids(opt_id=self.opt_id):
            world.set_new_context(ctx=ctx, singleton=False)
        else: 
            world.update_context(ctx=ctx)

        for opt in self._downstream:
            opt.run(world)

    def evaluate(self, world: World):
        pass

    def save_checkpoint(self):
        pass


class DummyEnv(Env):
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


class DummyOptimizer(Optimizer):
    def run(self, world: World) -> None:
        obs, _ = self.env.reset()
        action = self.env.action_space.sample()
        _, reward, _, _, _ = self.env.step(action)

        ctx = Context(
            id="reward_signal", 
            opt_id=self.id, 
            payload=SignalContext(value=reward)
        )
        
        if ctx.id not in world.get_opt_ctx_ids(opt_id=self.opt_id):
            world.set_new_context(ctx=ctx, singleton=False)
        else: 
            world.update_context(ctx=ctx)

    def evaluate(self, world: World):
        pass

    def save_checkpoint(self):
        pass


class DummyContextWrapper(ContextWrapper):
    def _get_violation_signal(self) -> float:
        ctx_ids = self._world.get_opt_ctx_ids(opt_id="parent")
        if not ctx_ids:
            return 0.0
        ctx = self._world.get_context(next(iter(ctx_ids)))
        return ctx.payload.value

    def _get_violation_penalty(self) -> float:
        return 1.0

    def observation(self, observation):
        return observation

    def action(self, action):
        return action

# TODO review this
class DummyOptimizerConfig(BaseOptimizerConfig):
    def __init__(self):
        super().__init__(opt_class=DummyOptimizer)

# TODO review this
class SuperDummyOptimizerConfig(BaseOptimizerConfig):
    def __init__(self):
        super().__init__(opt_class=SuperDummyOptimizer)

@pytest.mark.integration
def test_core_framework_end_to_end():
    # Setup Shared world
    world = World()

    # Setup Child Optimizer
    wrapped_env = DummyContextWrapper(
        env=DummyEnv(),
        world=world,
    )
    child_cfg = DummyOptimizerConfig().environment(wrapped_env)
    child = child_cfg.build_optimizer()
    child.set_id("child")
    world.register_optimizer(child)

    # Setup Meta Optimizer
    meta_cfg = SuperDummyOptimizerConfig()
    meta_opt = meta_cfg.build_optimizer()
    meta_opt.set_id("parent")
    meta_opt.set_downstream(opt=child)
    world.register_optimizer(meta_opt)
    
    # Build assertions
    assert world.get_ctx_ids() == set()
    assert world.get_opt_ids() == {"child", "parent"}
    
    # Run Meta Optimizer (will also run child)
    meta_opt.run(world)

    # World assertions
    ctx_ids = world.get_ctx_ids()
    assert len(ctx_ids) == 2, "Expected two contexts (regulation + reward)"

    parent_ctx_ids = world.get_opt_ctx_ids("parent")
    child_ctx_ids = world.get_opt_ctx_ids("child")

    assert parent_ctx_ids == {"regulation_signal"}
    assert child_ctx_ids == {"reward_signal"}

    # Context payload
    regulation_ctx = world.get_context("regulation_signal")
    reward_ctx = world.get_context("reward_signal")

    assert isinstance(regulation_ctx.payload, SignalContext)
    assert isinstance(reward_ctx.payload, SignalContext)

    assert regulation_ctx.payload.value == 1.0
    assert isinstance(reward_ctx.payload.value, float)

    # Env Wrapper assertions
    violation_signal = wrapped_env._get_violation_signal()
    assert violation_signal == 1.0, "Env should read regulation signal from world"
    
    # Run a downstream optimizer only
    prev_reward = reward_ctx.payload.value
    child.run(world)

    reward_ctx_updated = world.get_context("reward_signal")
    assert reward_ctx_updated.payload.value != prev_reward
    assert len(world.get_ctx_ids()) == 2, "Context count must remain stable"
