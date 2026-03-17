from dataclasses import Field
from typing import Optional, TypeAlias
from dataclasses import dataclass

from loggers.schemas import LoggerSchema

PolicyID: TypeAlias = str 
EnvID: TypeAlias = str

# TODO review this hierarchy

@dataclass
class TransitionStats:
    # TODO models such as PILCO, Dyna, Qyna-Q
    # statistics collected at env-step level
    action: float
    observation: float  # or state
    reward: float
    terminated: bool
    truncated: bool

    # calculated stats
    logp: Optional[float] = None
    value_pred: Optional[float] = None
    value_target: Optional[float] = None
    advantage: Optional[float] = None
    td_error: Optional[float] = None
    q_pred: Optional[float] = None

    
@dataclass
class LearnerStats(LoggerSchema):
    batch_size: int = Field(default=0, json_schema_extra={"reduce"})
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

PolicyStats: TypeAlias = dict[PolicyID, LearnerStats]
EnvStats: TypeAlias = dict[EnvID, TransitionStats]

@dataclass
class EpisodeStats:
    # Reward (R) statistics
    reward: Optional[float]
    reward_mean: Optional[float]
    reward_min: float
    reward_max: float
    reward_per_step: float
    reward_terminal: float

    # Value (V) statistics
    value_terminal: float
    value_penultimate: float

    # Trajectory info
    episode_len_mean: float
    episode_len_max: float
    episode_len_min: float

    # Learner statistics
    learner: PolicyStats = Field(default_factory=dict)

    # transition statistics
    env: EnvStats = Field(default_factory=dict)



class PPOStats(LoggerSchema):
    train_episode: EpisodeStats = Field(default_factory=dict)
    eval_episode: EpisodeStats = Field(default_factory=dict)



    # @dataclass
# class AggEpisodeStats:
#     episode_reward_mean: float
#     episode_reward_min: float
#     episode_reward_max: float
#     episode_len_mean: float  # accross different episodes
#     episode_len_max: float  # accross different episodes
#     episode_len_min: float


