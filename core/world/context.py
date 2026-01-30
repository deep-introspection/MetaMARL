from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, SkipValidation

from core.mechanism.base import Mechanism
from core.types import ContextID, OptimizerID


# TODO some world contexts are singletons (mutable) others are simply mutable.
# TODO for now singleton/or no is deffered to world
# TODO Enums for Context to access different Context Schemas.
class ContextSchema(BaseModel):
    """Base schema for shared world context."""

    model_config = {"arbitrary_types_allowed": True}


class MechanismContext(ContextSchema):
    env_id: str
    theta: SkipValidation[Mechanism]


# TODO strict type annotations rm Any
class EnvStepContext(ContextSchema):
    observation: Any
    reward: float | np.ndarray
    action: Any
    info: dict


@dataclass
class Context:
    """
    Runtime instance of a context
    """

    id: ContextID | None
    opt_id: OptimizerID
    step: int
    env: str
    payload: ContextSchema
