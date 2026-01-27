from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

from gymnasium.core import ObsType

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism
from core.optimizers.base import Optimizer
from core.world.base import World
from core.world.context import MechanismContext


class RegulatorEnv(BaseEnv):
    def __init__(
        self,
        *,
        world: World,
        optimizer: Optional[Optimizer] = None,
        train_iters: int = 5,
        eval_iters: int = 2,
        **kwargs,
    ):
        super().__init__(world=world)
        self.inner: Optimizer = optimizer
        self.train_iters: int = train_iters
        self.eval_iters: int = eval_iters

        self._validate()

    def _validate(self):
        if self.inner is None:
            return  # analytic override mode allowed

        if self.train_iters <= 0:
            raise ValueError("train_iters must be >= 1 when optimizer is provided")

        if self.eval_iters <= 0:
            raise ValueError("eval_iters must be >= 1 when optimizer is provided")

    @override(BaseEnv)
    def _step(
        self, theta: Mechanism
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.inner is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no inner optimizer. "
                f"Override `_step()` for analytic reward computation."
            )

        self._publish(MechanismContext(theta=theta))

        # Train inner optimizer
        for _ in range(self.train_iters):
            self.inner.run()
        return None, None, False, False, {}

    # @abstractmethod
    # @override(BaseEnv)
    # def observation(self, observation: ObsType) -> ObsType:
    #     # read downstream results from optimizer and compute aggregate
    #     raise NotImplementedError

    @abstractmethod
    def aggregate_rewards(self, rewards: list[SupportsFloat]) -> SupportsFloat:
        return NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        if self.inner is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no inner optimizer. "
                f"Override `reward()` for analytic reward computation."
            )

        # read downstream results from optimizer and compute aggregate reward
        # Evaluate inner optimizer
        rewards = [self.inner.evaluate() for _ in range(self.eval_iters)]
        return self.aggregate_rewards(rewards)

    # @abstractmethod
    # @override(BaseEnv)
    # def action(self, action: ActType) -> ActType:
    #     # read dowsntream results from optimzier and compute aggregate action
    #     raise NotImplementedError
