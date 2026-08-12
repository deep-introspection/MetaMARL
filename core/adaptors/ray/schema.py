from pydantic import Field
from typing import Optional, TypeAlias
from core.metrics.schemas import MetricSchema
from core.metrics.enums import ReduceProtocol

PolicyID: TypeAlias = str
EnvID: TypeAlias = str
AgentID: TypeAlias = str

class AgentEnvStepSchema(MetricSchema):
    """Generic metrics produced for one agent during environment steps."""
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

    # TODO is this necessary ?
    intrinsic_utility: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    violation_signal: float = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

# TODO add recducer metadata attachment.
class EnvStepSchema(MetricSchema):  # attention this is aggregate by env not by env step
    """Generic environment-step metrics."""
    # TODO models such as PILCO, Dyna, Qyna-Q
    by_agent: dict[AgentID, AgentEnvStepSchema] = Field(
        default_factory=dict,
    )


class PolicyLearnerSchema(MetricSchema):
    batch_size: int
    # Value, Q, advantage debugging
    total_loss: Optional[float] = Field(
            default=None,
            json_schema_extra={
                "source": "total_loss"
            }
        )
    residual_variance: Optional[float] = None

    # TODO this is a callable
    sample_staleness: Optional[float] = None

    # Policy (π) debugging
    policy_loss: float = Field(
        default=None,
        json_schema_extra={
            "source": "policy_loss",
            "reduce": ReduceProtocol.MEAN
        }
    )
    policy_entropy: float = Field(
        default=None,
        json_schema_extra={
            "source": "entropy",
            "reduce": ReduceProtocol.MEAN
        }
    )
    policy_entropy_coeff: float = Field(
        default=None,
        json_schema_extra={
            "source": "curr_kl_coeff",
            "reduce": ReduceProtocol.MEAN
        }
    )
    policy_relative_entropy: float  # inferred
    policy_kl: float = Field(
        default=None,
        json_schema_extra={
            "source": "kl",
            "reduce": ReduceProtocol.MEAN
        }
    )
    policy_kl_coeff: float = Field(
        default=None,
        json_schema_extra={
            "source": "curr_kl_coeff",
            "reduce": ReduceProtocol.MEAN
        }
    )
    # TODO what is total loss ?
    # TODO kl vs kl loss
    # TODO curr_kl_coeff
    # TODO entropy vs entropy coeff

    # Value (V) debgging
    value_loss: float = Field(
        default=None,
        json_schema_extra={
            "source": "vf_loss",
            "reduce": ReduceProtocol.MEAN
        }
    )
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


class EpisodeRolloutSchema(MetricSchema):  # Episode rollout = aggregate over env steps
    # Reward (R) statistics
    reward_total: Optional[float] = None
    reward_mean: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "source": "episode_return_mean",
            "reduce": ReduceProtocol.MEAN
        }
    )
    reward_min: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "source": "episode_return_min",
            "reduce": ReduceProtocol.MEAN
        }
    )
    reward_max: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "source": "episode_return_max",
            "reduce": ReduceProtocol.MEAN
        }
    )
    reward_per_step: Optional[float] = None
    reward_terminal: Optional[float] = None

    # Value (V) statistics
    value_terminal: Optional[float] = None
    value_penultimate: Optional[float] = None

    # Trajectory info
    episode_len_mean: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "source": "episode_len_mean",
            "reduce": ReduceProtocol.MEAN
        }
    )
    episode_len_max: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "source": "episode_len_min",
            "reduce": ReduceProtocol.MEAN
        }
    )
    episode_len_min: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "source": "episode_len_max",
            "reduce": ReduceProtocol.MEAN
        }
    )


class EnvRolloutSchema(MetricSchema):
    aggregate: EpisodeRolloutSchema
    step_series: EnvStepSchema

class RolloutSchema(MetricSchema):
    aggregate: EpisodeRolloutSchema
    by_policy: dict[PolicyID, EpisodeRolloutSchema] = Field(default_factory=dict)
    by_env: dict[EnvID, EnvRolloutSchema] = Field(default_factory=dict)


class LearnerSchema(MetricSchema):
    by_policy: dict[PolicyID, PolicyLearnerSchema] = Field(default_factory=dict)


class TrainSchema(MetricSchema):
    rollout: RolloutSchema = Field(json_schema_extra = {"source" : "env_runners"})
    learner: LearnerSchema = Field(json_schema_extra = {"source" : "learners"})


class EvalSchema(MetricSchema):
    rollout: RolloutSchema = Field(json_schema_extra = {"source" : "env_runners"})


class RaySchema(MetricSchema):
    train: TrainSchema = Field(
            default=None,
            json_schema_extra={
                "source" : "."
            }
        )
    eval: EvalSchema = Field(json_schema_extra={"source" : "evaluation"})


# TODO 
# num_env_steps_sampled_lifetime_throughput
# timers
# throughput_since_last_restore
# num_agent_steps_sampled
# num_agent_steps_sampled_lifetime
# prevent non terminal leaves to have reduce objects