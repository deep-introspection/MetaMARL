from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """Fitness score published by child optimizer after evaluation."""

    objective_score: float
    mean_reward: float
    collapse_rate: float
    sustainability_penalty: float
