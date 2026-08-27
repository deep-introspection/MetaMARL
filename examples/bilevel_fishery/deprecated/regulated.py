import logging

import numpy as np
from gymnasium.core import ActType

from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism
from core.utils import sigmoid
from core.world.context import MechanismContext

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

    def _normalize_action(
        self,
        action_component: ActType,
    ) -> ActType:
        z = np.asarray(action_component, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray(
            [sigmoid(float(value) / temperature) for value in z], dtype=np.float32
        )
