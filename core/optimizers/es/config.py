from typing import Optional, Self

from core.annotations import override
from core.optimizers.config import OptimizerConfig
from core.optimizers.es.optimizer import ESOptimizer


class ESConfig(OptimizerConfig):
    def __init__(self, opt_class=None):
        super().__init__(opt_class=opt_class or ESOptimizer)

        # Add default or from default
        # ES training hyperparameters
        self.dimension: int = None
        self.sigma: int = 0.15
        self.mean_lr: float = 0.1
        self.sigma_lr: float = 0.05
        self.min_sigma: float = 1e-3
        self.max_sigma: float = 0.5
        self.break_symmetry: bool = False

        self.convergence_eps: float = 1e-4
        self.convergence_patience: int = 10

    @override(OptimizerConfig)
    def training(
        self,
        *,
        sigma: Optional[int] = None,
        mean_lr: Optional[float] = None,
        sigma_lr: Optional[float] = None,
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
        generation: Optional[int] = None,
        break_symmetry: Optional[bool] = None,
        convergence_eps: Optional[float] = None,
        convergence_patience: Optional[int] = None,
        **kwargs,
    ) -> Self:
        """Sets the training related configs for Evolution Strategies (ES).

        Args:
            pop_size: Number of candidate solutions sampled per ES generation. Larger population
                provide more stable gradient estimates at the cost of increased computation.
            sigma: Initial standard deviation (std) of the search distribution.
            mean_lr: Learning rate for updating the mean of the search distribution which sets
                how aggressively the optimizer shifts the center of the distribution towards
                higer performing candidates.
            min_sigma: Lower bound on the std. Prevents premature convergence by ensuring a min
                level of exploration is maintained.
            max_sigma: Upper bound on the std. Prevents excessive large exploration steps that
                can destabilize training.

        Returns:
            This updated OptimizerConfig object.
        """
        super().training(**kwargs)

        if sigma is not None:
            self.sigma = sigma
        if mean_lr is not None:
            self.mean_lr = mean_lr
        if sigma_lr is not None:
            self.sigma_lr = sigma_lr
        if min_sigma is not None:
            self.min_sigma = min_sigma
        if max_sigma is not None:
            self.max_sigma = max_sigma
        if generation is not None:
            self.generation = generation
        if break_symmetry is not None:
            self.break_symmetry = break_symmetry
        if convergence_eps is not None:
            self.convergence_eps = convergence_eps
        if convergence_patience is not None:
            self.convergence_patience = convergence_patience

        return self

    # @override(OptimizerConfig)
    # def evaluation(
    #     self, *, evaluation_best_fitness, evaluation_best_candidate, **kwargs
    # ) -> Self:
    #     raise NotImplementedError

    # # TODO this is where the random seed goes
    # # TODO do we put rng here ?
    # @override(OptimizerConfig)
    # def fault_tolerance(self, *, rng: Optional[float] = None, **kwargs) -> Self:
    #     return super().fault_tolerance()
