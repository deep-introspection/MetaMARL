from src.es.optimizer import ESOptimizer
from core.optimizers.config import OptimizerConfig
from typing import Self, override, Optional


class ESConfig(OptimizerConfig):
    def __init__(self, opt_class=None):
        super().__init__(opt_class=opt_class or ESOptimizer)


        # ES specific settings (default)
        # Distribution and population
        self.dimension: int = 5
        self.pop_size: int = 8
        self.sigma: int = 0.15

        # Distribution lr
        self.mean_lr: float = 0.1
        self.sigma_lr: float = 0.05

        # sigma constraints
        self.min_sigma: float = 1e-3
        self.max_sigma: float = 0.5

        # Random number generator
        self.rng = np.random.default_rng(self.random_seed)
        
        # History tracking
        self.generation = 0
        self.best_fitness = -float("inf")
        self.best_candidate = self.mean.copy()

    @override(OptimizerConfig)
    def training(
        self,
        *,
        dimension: Optional[int] = None,
        pop_size: Optional[int] = None,
        sigma: Optional[int] = None,
        mean_lr: Optional[float] = None,
        sigma_lr: Optional[float] = None,
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
        **kwargs,
    ) -> Self:
        """Sets the training related configs for Evolution Strategies (ES).

        Args:
            dimenstion: Dimensionality of the search space which corresponds to the number of 
                parameters being optimized (e.g., number of mechanism params in the outer loop).
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

        if dimension is not None:
            self.dimension = dimension
        if pop_size is not None:
            self.pop_size = pop_size
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
        
        return self
    

    @override(OptimizerConfig)
    def evaluation(
        self,
        *,
        evaluation_best_fitness,
        evaluation_best_candidate,
        **kwargs
    ) -> Self : 
        raise NotImplementedError
    

    # TODO this is where the random seed goes
    @override(OptimizerConfig)
    def fault_tolerance(
        self,
        *,
        rng,
        **kwargs) -> Self:
        return super().fault_tolerance()
    
        if rns is not None:
            self.rng = rng
