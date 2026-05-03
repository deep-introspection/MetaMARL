"""Context schemas for the bilevel fishery experiment.

A *context* is a typed data container that carries information between the
inner optimizer (APPO) and the outer optimizer (ES).  Each context is
serialised by the inner loop at the end of an evaluation and consumed by the
outer loop to update the mechanism-space search.

Classes
-------
FitnessContext
    Scalar fitness signal and supporting metrics produced after one inner-loop
    evaluation and forwarded to the ES regulator environment.
"""

from typing import SupportsFloat

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """Fitness signal produced by the inner optimizer after one evaluation.

    Published by the inner APPO optimizer and consumed by the outer ES
    regulator environment.  This is the *sole* scalar feedback channel for
    bilevel optimization: the :attr:`objective_score` field is used directly
    as the ES fitness signal.

    Attributes
    ----------
    objective_score : float
        Scalar ES fitness: ``mean_reward - sustainability_weight *
        sustainability_penalty``.  Higher is better.
    mean_reward : float
        Mean per-step reward averaged over all agents and evaluation steps.
    collapse_rate : float
        Fraction of steps where the fish stock was below the sustainability
        threshold (0 = never collapsed, 1 = always collapsed).
    sustainability_penalty : float
        Normalised mean shortfall below the sustainability threshold.
    total_fines : float
        Cumulative fines paid across all agents during evaluation.
    mean_fish : float
        Mean normalised fish stock (in [0, 1]) over the evaluation trajectory.
    min_fish : float
        Minimum normalised fish stock observed during evaluation.
    """

    objective_score: float
    mean_reward: float
    collapse_rate: float
    sustainability_penalty: float
    total_fines: float
    mean_fish: float
    min_fish: float

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
    ) -> "FitnessContext":
        """Construct a :class:`FitnessContext` from raw evaluation metrics.

        The scalar objective combines economic welfare and sustainability:

        .. math::

            \\text{objective} = \\text{mean\\_reward}
                              - w_{\\text{sus}} \\cdot \\text{sustainability\\_penalty}

        where :math:`w_{\\text{sus}}` = ``sustainability_weight``.

        Parameters
        ----------
        mean_reward : SupportsFloat
            Mean per-step reward averaged over all agents and time steps in
            the evaluation episode(s).
        collapse_rate : SupportsFloat
            Fraction of time steps where the fish stock fell below the
            sustainability threshold.
        sustainability_penalty : SupportsFloat
            Normalised mean shortfall below the sustainability threshold,
            computed as ``mean(max(0, threshold - fish) / threshold)``.
        sustainability_weight : SupportsFloat
            Weight applied to the sustainability penalty in the objective
            (``sus_weight`` in the outer ecology config).
        total_fines : SupportsFloat, optional
            Cumulative fines paid by all agents during evaluation.
            Default is ``0.0``.
        mean_fish : SupportsFloat, optional
            Mean normalised fish stock over the evaluation trajectory.
            Default is ``0.0``.
        min_fish : SupportsFloat, optional
            Minimum normalised fish stock observed during evaluation.
            Default is ``0.0``.

        Returns
        -------
        FitnessContext
            Populated context carrying the computed ``objective_score`` and
            all supporting metrics.
        """

        objective = float(mean_reward - sustainability_weight * sustainability_penalty)

        return cls(
            objective_score=objective,
            mean_reward=float(mean_reward),
            collapse_rate=float(collapse_rate),
            sustainability_penalty=float(sustainability_penalty),
            total_fines=float(total_fines),
            mean_fish=float(mean_fish),
            min_fish=float(min_fish),
        )
