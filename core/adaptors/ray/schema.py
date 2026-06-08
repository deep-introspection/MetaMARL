from pydantic import Field
from typing import Optional, TypeAlias
from core.reporting.loggers.enums import ReduceProtocol

from core.reporting.loggers.schemas import LoggerSchema

PolicyID: TypeAlias = str
EnvID: TypeAlias = str


# TODO add recducer metadata attachment.
class EnvStepSchema(LoggerSchema):  # attention this is aggregate by env not by env step
    # TODO models such as PILCO, Dyna, Qyna-Q
    # statistics collected at env-step level
    action: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )  # TODO by agentID
    observation: float = Field(json_schema_extra={"reduce": ReduceProtocol.SERIES})
    reward: float = Field(json_schema_extra={"reduce": ReduceProtocol.SERIES})
    terminated: bool = Field(json_schema_extra={"reduce": ReduceProtocol.SERIES})
    truncated: bool = Field(json_schema_extra={"reduce": ReduceProtocol.SERIES})

    # calculated stats
    logp: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    value_pred: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    value_target: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    advantage: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    td_error: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    q_pred: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )


class PolicyLearnerSchema(LoggerSchema):
    batch_size: int
    # Value, Q, advantage debugging
    total_loss: Optional[float] = None
    residual_variance: Optional[float] = None
    sample_staleness: Optional[float] = None

    # Policy (π) debugging
    policy_loss: float
    policy_entropy: float
    policy_entropy_coeff: float
    policy_relative_entropy: float  # inferred
    policy_kl: float
    policy_kl_coeff: float
    # TODO what is total loss ?
    # TODO kl vs kl loss
    # TODO curr_kl_coeff
    # TODO entropy vs entropy coeff

    # Value (V) debgging
    value_loss: float
    value_mean: float
    value_target: float

    # TODO Q-statistics
    # TODO Advantage statistics
    # TODO advantage statistics

    # TODO Reward (R) debugging
    # Reward metrics. N.B. episode == trajectory

    # Gradient debugging
    gradient_norm: float
    gradient_noise: float


class EpisodeRolloutSchema(LoggerSchema):  # Episode rollout = aggregate over env steps
    # Reward (R) statistics
    reward_total: Optional[float] = None
    reward_mean: Optional[float]
    reward_min: float
    reward_max: float
    reward_per_step: float
    reward_terminal: float

    # Value (V) statistics
    value_terminal: Optional[float] = None
    value_penultimate: Optional[float] = None

    # Trajectory info
    episode_len_mean: float
    episode_len_max: float
    episode_len_min: float


class EnvRolloutSchema(LoggerSchema):
    aggregate: EpisodeRolloutSchema
    step_series: EnvStepSchema



class RolloutSchema(LoggerSchema):
    aggregate: EpisodeRolloutSchema
    by_env: dict[EnvID, EnvRolloutSchema] = Field(default_factory=dict)


class LearnerSchema(LoggerSchema):
    by_policy: dict[PolicyID, PolicyLearnerSchema] = Field(default_factory=dict)


class TrainSchema(LoggerSchema):
    rollout: RolloutSchema
    learner: LearnerSchema


class EvalSchema(LoggerSchema):
    rollout: RolloutSchema


class RaySchema(LoggerSchema):
    train: TrainSchema
    eval: EvalSchema
