import logging
from abc import abstractmethod
from typing import Optional, SupportsFloat

import numpy as np
import ray

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.types import OptimizerID
from core.world.base import World
from core.world.context import MechanismContext, MechanismStatus

from pathlib import Path
import os
import json

logger = logging.getLogger(__name__)


class RegulatedEnv(BaseEnv):
    def __init__(
        self,
        *,
        mechanism_id: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        # mechanism_space can be a class or an instance
        self.mechanism_id = mechanism_id
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        
        self._using_default_mechanism = True

    @property
    def mechanism(self) -> Mechanism:
        if self.m is not None:
            return self.m
        return self.m_space.default()
    
    @property
    def published_mechanism_assigned(self) -> bool:
        return self.m is not None and not self._using_default_mechanism
    
    @override(BaseEnv)
    def _pre_reset(self, seed: Optional[int] = None):
        # Try to fetch a new mechanism if one is available (published)
        # Otherwise keep the current mechanism for subsequent episodes
        if self.mechanism_id is None:
            raise RuntimeError(
                "RegulatedEnv has no mechanism_id. "
                "mechanism_id must be injected at env creation."
            )

        if not self.published_mechanism_assigned: 
            try:
                new_ctx = ray.get(
                    self.world.get_mechanism_by_id.remote(
                        mechanism_id = self.mechanism_id, 
                        seed=self.seed,
                        mode=self.mode
                    )
                )
            except Exception as e:
                self._debug_remote(
                    "pre_reset_fetch_failed",
                    {
                        "error_type": type(e).__name__,
                        "error_repr": repr(e),
                    },
                )
                raise RuntimeError(
                    f"Could not fetch mechanism_id={self.mechanism_id} from World."
                ) from e

            if new_ctx is not None:
                self.m_ctx = new_ctx
                self.m = self.m_ctx.mechanism
                self._using_default_mechanism = False

            # TODO raising error if training started and default mechanism is still on - leads to silent error

    @abstractmethod
    def violation_signal(self, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    def penalty(self, **kwargs) -> float:
        raise NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat, **kwargs) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`en v` :meth:`step` reward

        Returns:
            The modified `reward`
        """
        return reward - self.penalty(**kwargs) * self.violation_signal(**kwargs)
