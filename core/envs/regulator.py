from abc import abstractmethod
from typing import Any, SupportsFloat

from gymnasium.core import ActType, ObsType

from core.annotations import override
from core.envs.base import BaseEnv
from core.optimizers.base import Optimizer
from core.world.base import World


class RegulatorEnv(BaseEnv):
    def __init__(self, world: World, optimizer: Optimizer, iters: int = 1):
        super().__init__(world=world)
        self.optimizer: Optimizer = optimizer
        self.iters: int = iters

    @override(BaseEnv)
    def _step(
        self, action: ActType | None = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]] | None:
        for _ in range(self.iters):
            self.optimizer.run()

    @abstractmethod
    @override(BaseEnv)
    def observation(self, observation: ObsType) -> ObsType:
        # read downstream results from optimizer and compute aggregate
        raise NotImplementedError

    @abstractmethod
    @override(BaseEnv)
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        # read downstream results from optimizer and compute aggregate reward
        raise NotImplementedError

    @abstractmethod
    @override(BaseEnv)
    def action(self, action: ActType) -> ActType:
        # read dowsntream results from optimzier and compute aggregate action
        raise NotImplementedError
