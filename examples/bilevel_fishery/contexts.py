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
    collapse_rate: float
    sustainability_penalty: float
    total_fines: float
    mean_fish: float
    min_fish: float
    mean_realized_harvest: float
    harvest_score: float

    @classmethod
    def from_metrics(
        cls,
        *,
        mean_reward: SupportsFloat,
        collapse_rate: SupportsFloat,
        sustainability_penalty: SupportsFloat,
        sustainability_weight: SupportsFloat,
        total_fines: SupportsFloat = 0.0,
        mean_fish: SupportsFloat = 0.0,
        min_fish: SupportsFloat = 0.0,
        mean_realized_harvest: SupportsFloat = 0.0,
        harvest_score: SupportsFloat = 0.0,
    ) -> "FitnessContext":
        """
        Construct fitness context from evaluation metrics.
        """
        # TODO
        # objective = mean_reward - sustainability_weight * (1.0 - mean_fish)
        # objective = harvest_score
        objective = harvest_score + sustainability_weight * mean_fish
        # objective = mean_fish

        # reward = float(mean_reward)
        # fish = float(mean_fish)
        # alpha = float(sustainability_weight)

        # if not 0.0 <= alpha <= 1.0:
        #     raise ValueError(
        #         "sustainability_weight must be between 0 and 1"
        #     )

        # eps = 1e-8

        # objective = (
        #     (1.0 - alpha) * np.log(max(reward, eps))
        #     + alpha * np.log(max(fish, eps))
        # )

        return cls(
            objective_score=objective,
            mean_reward=float(mean_reward),
            collapse_rate=float(collapse_rate),
            sustainability_penalty=float(sustainability_penalty),
            total_fines=float(total_fines),
            mean_fish=float(mean_fish),
            min_fish=float(min_fish),
            mean_realized_harvest=float(mean_realized_harvest),
            harvest_score=float(harvest_score),
        )
