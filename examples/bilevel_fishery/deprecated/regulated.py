import logging
from abc import abstractmethod
from typing import Optional, SupportsFloat

import numpy as np
import ray
from gymnasium.core import ActType, ObsType

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace
from core.types import OptimizerID
from core.utils import sigmoid
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

   

    def _normalize_action(
        self,
        action_component: ActType,
    ) -> ActType:
        z = np.asarray(action_component, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray([sigmoid(float(value) / temperature) for value in z], dtype=np.float32)