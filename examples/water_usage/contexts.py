from typing import SupportsFloat

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """
    Fitness signal produced by inner optimizer after evaluation.
    """

    objective_score: float
    mean_reward: float
    collapse_rate: float
    sustainability_penalty: float

    @classmethod
    def from_metrics(
        cls,
        *,
        mean_reward: SupportsFloat,
        collapse_rate: SupportsFloat,
        sustainability_penalty: SupportsFloat,
        sustainability_weight: SupportsFloat,
    ) -> "FitnessContext":
        objective = float(mean_reward - sustainability_weight * sustainability_penalty)

        return cls(
            objective_score=objective,
            mean_reward=float(mean_reward),
            collapse_rate=float(collapse_rate),
            sustainability_penalty=float(sustainability_penalty),
        )
