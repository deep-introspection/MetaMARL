import logging
from abc import abstractmethod
from typing import Optional, SupportsFloat

import ray

from core.annotations import override
from core.envs.base import BaseEnv
from core.types import OptimizerID
from core.world.base import World
from core.world.context import MechanismContext, MechanismStatus

logger = logging.getLogger(__name__)


class RegulatedEnv(BaseEnv):
    def __init__(
        self,
        *,
        world: World,
        opt_id: OptimizerID | None = None,
        **kwargs,
    ) -> None:
        super().__init__(world=world, opt_id=opt_id, **kwargs)

    @override(BaseEnv)
    def _pre_reset(self):
        # Try to fetch a new mechanism if one is available (published)
        # Otherwise keep the current mechanism for subsequent episodes
        try:
            new_ctx = ray.get(self.world.try_get_mechanism.remote())
        except Exception:
            new_ctx = None

        if new_ctx is not None:
            self.m_ctx = new_ctx
            self.m = self.m_ctx.mechanism
        # Fallback: if no mechanism yet, we must get one

        # TODO better fallback mechanism since this leads to silent bugs!
        if self.m_ctx is None or self.m is None:
            try:
                self.m_ctx = ray.get(self.world.get_mechanism.remote())
                self.m = self.m_ctx.mechanism
            except RuntimeError:
                # fallback baseline mechanism (must exist locally)
                self.m_ctx = MechanismContext(
                    index=-1,
                    env_id=None,
                    mechanism=self.m_space.default(),
                    status=MechanismStatus.init,
                    job=None,
                    metrics=None,
                )
                self.m = self.m_ctx.mechanism

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
        return reward - self.penalty(reward) * self.violation_signal(reward)
