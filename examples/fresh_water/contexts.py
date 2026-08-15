from typing import SupportsFloat

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """
    Fitness signal produced by inner optimizer after evaluation.

    Published by:
        inner optimizer (PPO)

    Consumed by:
        regulator environment (ES outer loop)

    This is the *sole* scalar feedback channel for bilevel optimization.
    """

    objective_score: float
    mean_reward: float
    crop_satisfaction: float
    streamflow_deviation: float
    streamflow_score: float

    @classmethod
    def from_metrics(
        cls,
        *,
        mean_reward: SupportsFloat,
        sustainability_weight: SupportsFloat,
        crop_satisfaction: SupportsFloat,
        streamflow_deviation: SupportsFloat,
    ) -> "FitnessContext":
        """
        Construct fitness context from evaluation metrics.
        """

        # objective = float(mean_reward - sustainability_weight * sustainability_penalty)
        streamflow_score = 1.0 / (1.0 + streamflow_deviation)
        objective = crop_satisfaction + sustainability_weight * streamflow_score

        return cls(
            objective_score=float(objective),
            mean_reward=float(mean_reward),
            crop_satisfaction=float(crop_satisfaction),
            streamflow_deviation=float(streamflow_deviation),
            streamflow_score=float(streamflow_score),
        )
