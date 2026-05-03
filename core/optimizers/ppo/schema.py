from dataclasses import Field
from typing import Optional, TypeAlias

from loggers.schemas import LoggerSchema

PolicyID: TypeAlias = str 
EnvID: TypeAlias = str
StepID: TypeAlias = int

# TODO add recducer identity
class EnvStepSchema(LoggerSchema): #attention this is aggregate by env not by env step
    """Schema for statistics recorded at the level of a single environment step.

    Although the class is named *step*, in practice it is aggregated across all
    steps within one environment rollout (see ``EnvRolloutSchema``).

    Attributes
    ----------
    action : float
        Action taken by the agent at this step.
    observation : float
        Observation (or state) received by the agent.
    reward : float
        Scalar reward signal returned by the environment.
    terminated : bool
        Whether the episode ended due to a terminal state.
    truncated : bool
        Whether the episode ended due to a time limit or truncation condition.
    logp : float, optional
        Log-probability of the taken action under the current policy.
    value_pred : float, optional
        Value function estimate :math:`V(s_t)` produced by the critic.
    value_target : float, optional
        TD target (bootstrapped return) used to train the value function.
    advantage : float, optional
        Generalised Advantage Estimate (GAE) :math:`\\hat{A}_t`.
    td_error : float, optional
        Temporal-difference error :math:`\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)`.
    q_pred : float, optional
        Q-value estimate (reserved for actor-critic variants with explicit Q).
    """

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
    
class PolicyLearnerSchema(LoggerSchema):
    """Learner-side statistics for a single PPO policy update step.

    Captures losses, entropy, KL divergence, value targets, and gradient
    diagnostics that are useful for monitoring PPO training stability and
    diagnosing policy collapse or reward hacking.

    Attributes
    ----------
    batch_size : int
        Number of transitions used in this update batch.
    total_loss : float, optional
        Combined scalar loss (policy + value + entropy bonus).
    residual_variance : float, optional
        Fraction of return variance not explained by the value baseline.
        Ideal value is close to 0; high values indicate a weak critic.
    sample_staleness : float, optional
        Mean number of update steps that have elapsed since each sample was
        collected.  Relevant for APPO where samples may be off-policy.
    policy_loss : float
        PPO clipped surrogate objective loss
        :math:`-L^{\\text{CLIP}}(\\theta)`.
    policy_entropy : float
        Mean entropy :math:`H[\\pi(\\cdot|s)]` of the current policy.
        High entropy encourages exploration; should not collapse to zero.
    policy_entropy_coeff : float
        Entropy regularisation coefficient applied to the loss.
    policy_relative_entropy : float
        Entropy of the current policy relative to a reference (inferred).
    policy_kl : float
        Mean KL divergence
        :math:`D_{\\text{KL}}[\\pi_{\\text{old}} \\| \\pi_\\theta]` between
        the old and new policies.  Used to monitor update magnitude.
    policy_kl_coeff : float
        Adaptive KL penalty coefficient (used in KL-penalised PPO variants).
    value_loss : float
        Value function regression loss
        :math:`\\mathbb{E}_t[(V_\\theta(s_t) - V_t^{\\text{target}})^2]`.
    value_mean : float
        Mean value function prediction :math:`\\mathbb{E}_t[V_\\theta(s_t)]`.
    value_target : float
        Mean TD target used to train the value function.
    gradient_norm : float
        L2 norm of the policy gradient before clipping.  Spikes may indicate
        instability.
    gradient_noise : float
        Estimated noise-to-signal ratio of the gradient.
    """

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


class EpisodeRolloutSchema(LoggerSchema): # Episode rollout = aggregate over env steps
    """Statistics aggregated over all environment steps within a single episode.

    Attributes
    ----------
    reward_total : float, optional
        Sum of rewards over the full episode.
    reward_mean : float, optional
        Mean per-step reward.
    reward_min : float
        Minimum per-step reward observed during the episode.
    reward_max : float
        Maximum per-step reward observed during the episode.
    reward_per_step : float
        Average reward per environment step (identical to ``reward_mean``
        when episodes have fixed length).
    reward_terminal : float
        Reward received at the terminal step (may include shaped bonuses).
    value_terminal : float, optional
        Value function estimate at the final state.  Useful for checking
        bootstrap accuracy in truncated episodes.
    value_penultimate : float, optional
        Value function estimate at the penultimate state.
    episode_len_mean : float
        Mean episode length across all episodes in the batch.
    episode_len_max : float
        Maximum episode length across all episodes in the batch.
    episode_len_min : float
        Minimum episode length across all episodes in the batch.
    """

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
    """Rollout statistics for a single environment instance.

    Combines per-episode aggregate statistics with per-step breakdowns,
    allowing fine-grained analysis of individual environment trajectories.

    Attributes
    ----------
    aggregate : EpisodeRolloutSchema
        Episode-level aggregate statistics (rewards, episode lengths, etc.).
    by_step : dict[StepID, EnvStepSchema]
        Mapping from step index to per-step observation/action/reward data.
    """

    aggregate: EpisodeRolloutSchema
    by_step: dict[StepID, EnvStepSchema]


class RolloutSchema(LoggerSchema):
    """Rollout statistics aggregated across all environment instances.

    Attributes
    ----------
    aggregate : EpisodeRolloutSchema
        Episode-level statistics averaged over all environments in the batch.
    by_env : dict[EnvID, EnvRolloutSchema]
        Per-environment rollout breakdowns, keyed by environment identifier.
        Defaults to an empty dict.
    """

    aggregate: EpisodeRolloutSchema
    by_env: dict[EnvID, EnvRolloutSchema] = Field(default_factory=dict)


class LearnerSchema(LoggerSchema):
    """Learner statistics for a single PPO training step.

    Attributes
    ----------
    by_policy : dict[PolicyID, PolicyLearnerSchema]
        Per-policy learner diagnostics (losses, KL, gradients, etc.),
        keyed by policy identifier.  Defaults to an empty dict.
    """

    by_policy: dict[PolicyID, PolicyLearnerSchema] = Field(default_factory=dict)


class TrainSchema(LoggerSchema):
    """Top-level schema for one PPO training iteration.

    Attributes
    ----------
    rollout : RolloutSchema
        Rollout statistics collected during sample generation.
    learner : LearnerSchema
        Learner diagnostics collected during the gradient update step.
    """

    rollout: RolloutSchema
    learner: LearnerSchema


class EvalSchema(LoggerSchema):
    """Top-level schema for one PPO evaluation run.

    Attributes
    ----------
    rollout : RolloutSchema
        Rollout statistics collected during evaluation (no gradient updates).
    """

    rollout: RolloutSchema


class PPOStats(LoggerSchema):
    """Root statistics schema for a full PPO training + evaluation cycle.

    Attributes
    ----------
    train : TrainSchema
        Statistics from the training phase of the current iteration.
    eval : EvalSchema
        Statistics from the evaluation phase of the current iteration.
    """

    train: TrainSchema
    eval: EvalSchema
