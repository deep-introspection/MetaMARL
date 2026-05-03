from dataclasses import dataclass
from enum import Enum
from typing import Optional, SupportsFloat

from gymnasium.core import ActType, ObsType
from pydantic import BaseModel, SkipValidation
from ray.rllib.utils.typing import MultiAgentDict

from core.mechanism.base import Mechanism
from core.types import ContextID, OptimizerID


class MechanismStatus(Enum):
    """Lifecycle states of a mechanism in the bilevel training pipeline.

    Attributes
    ----------
    init : str
        Mechanism has been created but not yet published to the World.
    published : str
        Mechanism is available for inner-loop environments to claim.
    assigned : str
        Mechanism has been claimed by an environment and is being used for
        training.
    train : str
        Mechanism is the active training target for the inner optimizer.
    eval : str
        Mechanism is being evaluated (policy is frozen, collecting rollouts).
    done : str
        Mechanism evaluation is complete; results are ready for aggregation.
    """

    init = "init"
    published = "published"
    assigned = "assigned"
    train = "train"
    eval = "eval"
    done = "done"


# TODO some world contexts are singletons (mutable) others are simply mutable.
# TODO for now singleton/or no is deffered to world
# TODO Enums for Context to access different Context Schemas.
class ContextSchema(BaseModel):
    """Base Pydantic schema for all payloads stored in the shared World.

    All context payload types must inherit from this class.  Setting
    ``arbitrary_types_allowed = True`` permits non-Pydantic objects (e.g.
    NumPy arrays, Gymnasium spaces) to be stored as field values.
    """

    model_config = {"arbitrary_types_allowed": True}


class MechanismContext(ContextSchema):
    """Context payload describing a mechanism candidate in the outer-loop pipeline.

    Published by :class:`~core.envs.regulator.RegulatorEnv` to the shared
    ``World`` actor so that inner-loop environments can claim and train under
    this mechanism.

    Attributes
    ----------
    index : int
        Ordinal position of this mechanism in the current ES population batch.
        ``-1`` indicates a fallback default mechanism.
    env_id : str or None
        Identifier of the environment that owns or last claimed this mechanism.
        ``None`` until assigned.
    status : MechanismStatus
        Current lifecycle state (see :class:`MechanismStatus`).
    job : MechanismStatus or None
        The phase this mechanism is being used for (``train`` or ``eval``).
    mechanism : Mechanism
        The typed mechanism object (e.g. quota, fine, ban parameters).
        Pydantic validation is skipped for this field because ``Mechanism``
        is a structural Protocol.
    metrics : ContextSchema or None
        Optional evaluation metrics attached after the evaluation phase.
    """

    index: int
    env_id: Optional[str]
    status: MechanismStatus
    job: Optional[MechanismStatus]
    mechanism: SkipValidation[Mechanism]
    metrics: Optional[ContextSchema]


# TODO strict type annotations rm Any
class EnvStepContext(ContextSchema):
    """Context payload capturing one environment step (or reset).

    Published by :class:`~core.envs.base.BaseEnv` (and its multi-agent
    variant) after every :meth:`step` call, and once after each :meth:`reset`
    (with ``reward=0.0`` and ``action=None``).  Used by
    :class:`~core.envs.regulator.RegulatorEnv` to aggregate fitness signals
    and by the W&B reporter for live experiment visualisation.

    Attributes
    ----------
    mechanism : int or None
        Index of the active mechanism (as stored in
        :class:`MechanismContext.index`).  ``None`` if no mechanism was active.
    observation : ObsType or MultiAgentDict
        Post-processed observation returned to the agent(s).
    observation_map : list[str] or None
        Ordered list of feature names corresponding to the observation vector
        dimensions.  Used for interpretable logging.
    reward : SupportsFloat or MultiAgentDict or list[float]
        Scalar reward (single-agent) or per-agent reward mapping (multi-agent).
    action : ActType or MultiAgentDict
        Action(s) submitted by the agent(s) during this step.  ``None`` on
        reset.
    info : dict or MultiAgentDict or None
        Auxiliary diagnostic information returned by the environment.
    """

    mechanism: Optional[int]
    observation: ObsType | MultiAgentDict
    observation_map: Optional[list[str]]
    reward: SupportsFloat | MultiAgentDict | list[float]
    action: ActType | MultiAgentDict
    info: dict | MultiAgentDict | None


@dataclass
class Context:
    """Runtime envelope wrapping a typed context payload.

    Produced by :meth:`~core.envs.base.BaseEnv._publish` and stored in the
    ``World`` actor.  The ``id`` is ``None`` until assigned by the World.

    Attributes
    ----------
    id : ContextID or None
        Unique identifier assigned by :meth:`~core.world.base.World.append_context`.
        ``None`` before registration.
    opt_id : OptimizerID
        Identifier of the optimizer that produced this context.
    step : int
        Environment timestep at which this context was created.
    env : str
        Class name of the environment that published this context
        (e.g. ``"FisheryEnv"``).
    payload : ContextSchema
        Typed payload — either an :class:`EnvStepContext` (per-step data) or a
        :class:`MechanismContext` (mechanism lifecycle data).
    """

    id: ContextID | None
    opt_id: OptimizerID
    step: int
    env: str
    payload: ContextSchema
