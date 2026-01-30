from abc import abstractmethod
from typing import Optional, SupportsFloat

from core.annotations import override
from core.envs.base import BaseEnv
from core.types import OptimizerID
from core.world.base import World


class RegulatedEnv(BaseEnv):
    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        **kwargs,
    ) -> None:
        super().__init__(world=world, opt_id=opt_id, **kwargs)

    @abstractmethod
    def violation_signal(self, reward: Optional[SupportsFloat] = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def penalty(self, reward: Optional[SupportsFloat] = None) -> float:
        raise NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        return reward - self.violation_penalty(reward) * self.violation_signal(reward)
