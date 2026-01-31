from dataclasses import dataclass
from enum import Enum
from typing import Optional, SupportsFloat

from gymnasium.core import ActType, ObsType
from pydantic import BaseModel, SkipValidation
from ray.rllib.utils.typing import MultiAgentDict

from core.mechanism.base import Mechanism
from core.types import ContextID, OptimizerID


class MechanismStatus(Enum):
    published = "published"
    assigned = "assigned"
    train = "train"
    eval = "eval"
    done = "done"


# TODO some world contexts are singletons (mutable) others are simply mutable.
# TODO for now singleton/or no is deffered to world
# TODO Enums for Context to access different Context Schemas.
class ContextSchema(BaseModel):
    """Base schema for shared world context."""

    model_config = {"arbitrary_types_allowed": True}


class MechanismContext(ContextSchema):
    index: int
    env_id: Optional[str]
    status: MechanismStatus
    job: Optional[MechanismStatus]
    mechanism: SkipValidation[Mechanism]
    metrics: Optional[ContextSchema]


# TODO strict type annotations rm Any
class EnvStepContext(ContextSchema):
    mechanism: Optional[int]
    observation: ObsType | MultiAgentDict
    reward: SupportsFloat | MultiAgentDict | list[float]
    action: ActType | MultiAgentDict
    info: dict | MultiAgentDict | None


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
