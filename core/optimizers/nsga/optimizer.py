import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from typing import Optional

class NGSAOptimizer:
    """
    Simple NSGA-II style optimizer for multi-objective fitness.
    Tracks population history and Pareto front.
    TODO: need to modify regulator_env to return matrix instead of list with fitness values
    """

    def __init__(self, config):
        # --- hyperparameters ---
        self.dimension: int = config.dimension
        self._batch_capacity: int = getattr(config, "batch_capacity", 32)
        self.rng = np.random.default_rng(getattr(config, "seed", None))

        # --- runtime state ---
        self.generation: int = 0
        self.best_fitness: Optional[np.ndarray] = None        # [sustainability, market_value]
        self.best_candidate: Optional[np.ndarray] = None      # parameters of best candidate
        self.population_history: list[tuple[np.ndarray, np.ndarray]] = []

        # Optional convergence tracking
        self.no_improve_steps: int = 0
        self.convergence_patience: int = getattr(config, "convergence_patience", 10)
        self.convergence_eps: float = getattr(config, "convergence_eps", 1e-4)
        self.converged_once: bool = False

        # Placeholder for environment
        self.env = None

    def environment(self, env, env_config=None):
        """
        Set the environment to use.
        """
        self.env = env(**(env_config or {}))
        return self

    @property
    def batch_capacity(self) -> int:
        return self._batch_capacity

    @batch_capacity.setter
    def batch_capacity(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_capacity must be positive")
        self._batch_capacity = value

    def _sample_population(self) -> np.ndarray:
        """
        Sample a population uniformly in [0,1]^dimension
        """
        return self.rng.random((self._batch_capacity, self.dimension), dtype=np.float32)

    def run(self) -> dict:
        if self.env is None:
            raise RuntimeError("Optimizer requires an environment")

        # Sample a population
        population = self._sample_population()  # (pop_size, dimension)

        # Step environment to get multi-objective fitness
        # Expected shape: (pop_size, 2) -> [sustainability, market_value]
        _, fitness_matrix, _, _, _ = self.env.step(population)
        fitness_matrix = np.asarray(fitness_matrix, dtype=np.float32)

        if fitness_matrix.size == 0:
            return {
                "converged": False,
                "best_fitness": None,
                "best_trajectory": None,
                "population_history": self.population_history,
            }

        # Store population and fitness for history
        self.population_history.append((population.copy(), fitness_matrix.copy()))

        # Compute Pareto front
        nds = NonDominatedSorting()
        front_indices = nds.do(fitness_matrix, only_non_dominated_front=True)
        pareto_solutions = population[front_indices]
        pareto_fitness = fitness_matrix[front_indices]

        # Pick representative best candidate (sum of objectives)
        best_idx = np.argmax(fitness_matrix.sum(axis=1))
        self.best_candidate = population[best_idx].copy()
        self.best_fitness = fitness_matrix[best_idx]

        # Get best trajectory if available
        best_trajectory = None
        if hasattr(self.env, "trajectories") and self.best_candidate is not None:
            best_trajectory = self.env.trajectories.get(best_idx)

        # Optional convergence logic
        # (implement if desired based on improvements over generations)
        converged = False

        return {
            "converged": converged,
            "best_fitness": self.best_fitness,
            "best_trajectory": best_trajectory,
            "population_history": self.population_history,
        }
