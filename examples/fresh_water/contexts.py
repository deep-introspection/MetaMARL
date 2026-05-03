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

    @classmethod
    def from_metrics(
        cls,
        *,
        mean_reward: SupportsFloat,
        collapse_rate: SupportsFloat,
        sustainability_penalty: SupportsFloat,
        sustainability_weight: SupportsFloat,
    ) -> "FitnessContext":
        """Construct a :class:`FitnessContext` from raw evaluation metrics.

        The scalar objective combines agent performance with a sustainability
        penalty:

        .. code-block:: text

            objective = mean_reward - sustainability_weight * sustainability_penalty

        Parameters
        ----------
        mean_reward : SupportsFloat
            Mean step-level reward collected by inner-loop agents under this
            mechanism.
        collapse_rate : SupportsFloat
            Fraction of timesteps where the resource stock fell below the
            sustainability threshold (in ``[0, 1]``).
        sustainability_penalty : SupportsFloat
            Mean normalized depth of sustainability-threshold violations;
            zero when the resource is always above the threshold.
        sustainability_weight : SupportsFloat
            Scalar weight controlling the trade-off between agent reward and
            ecosystem sustainability in the objective.

        Returns
        -------
        FitnessContext
            Populated fitness context with the computed objective score.
        """

        objective = float(mean_reward - sustainability_weight * sustainability_penalty)

        return cls(
            objective_score=objective,
            mean_reward=float(mean_reward),
            collapse_rate=float(collapse_rate),
            sustainability_penalty=float(sustainability_penalty),
        )
