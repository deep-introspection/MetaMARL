from abc import abstractmethod
from typing import SupportsFloat

from core.annotations import override
from core.envs.base import BaseEnv
from core.types import OptimizerID
from core.world.base import World


class RegulatedEnv(BaseEnv):
    def __init__(
        self, *, world: World, opt_id: OptimizerID | None = None, **kwargs
    ) -> None:
        super().__init__(world=world, opt_id=opt_id, **kwargs)

    @abstractmethod
    def violation_signal(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def violation_penalty(self) -> float:
        raise NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        return reward - self.violation_penalty() * self.violation_signal()
