import math

import numpy as np

from core.optimizers.base import Optimizer
from src.es.config import ESConfig

# TODO move this into constants
EPS = 1e-8


class ESOptimizer(Optimizer):
    def __init__(self, config: ESConfig):
        super().__init__(config)

        # --- hyperparameters --
        self.dimension = config.dimension
        self.population_size = config.pop_size
        self.mean_lr = config.mean_lr
        self.sigma_lr = config.sigma_lr
        self.min_sigma = config.min_sigma
        self.max_sigma = config.max_sigma

        # --- runtime state ---
        # Initialize search distribution at center of unit cube
        self.mean = np.full(config.dimension, 0.5, dtype=np.float32)
        self.sigma = float(config.sigma)

        # Random number generator
        self.rng = np.random.default_rng(config.seed)

        # History tracking
        self.generation = 0
        self.best_fitness = -float("inf")
        self.best_candidate = self.mean.copy()

    # TODO refactor this to Env_Runner sampler
    def _sample_population(self) -> np.ndarray:
        """Sample population for current generation using antithetic sampling.

        Uses logit reparametrization to avoid boundary clipping and
        antithetic sampling to reduce variance.

        Returns:
            Population matrix of shape (population_size, dimension)
        """
        # Ensure even population size for antithetic sampling
        half_pop = self.population_size // 2
        remaining = self.population_size - (2 * half_pop)

        # Sample noise for half the population
        noise_half = self.rng.standard_normal(
            (half_pop, self.dimension), dtype=np.float32
        )

        # Create antithetic pairs (mirrored noise)
        if half_pop > 0:
            noise_matrix = np.vstack([noise_half, -noise_half])
        else:
            noise_matrix = np.empty((0, self.dimension), dtype=np.float32)

        # Add remaining samples if population size is odd
        if remaining > 0:
            extra_noise = self.rng.standard_normal(
                (remaining, self.dimension), dtype=np.float32
            )
            noise_matrix = np.vstack([noise_matrix, extra_noise])

        # Transform mean to logit space for unbounded optimization
        # logit(p) = log(p/(1-p)), inverse_logit(x) = 1/(1+exp(-x))
        eps_bound = 1e-6
        mean_clipped = np.clip(self.mean, eps_bound, 1.0 - eps_bound)
        mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))

        # Add noise in logit space
        population_logit = mean_logit[None, :] + self.sigma * noise_matrix

        # Transform back to probability space using sigmoid
        population = 1.0 / (1.0 + np.exp(-population_logit))

        return population.astype(np.float32)

    # TODO refactor this into Learner
    def _update_parameters(
        self, population: np.ndarray, fitness_scores: list[float]
    ) -> None:
        """Update ES distribution parameters using fitness-weighted gradients.

        Uses rank-based fitness shaping for robustness to outliers.
        Works in logit space for unconstrained optimization.

        Args:
            population: Population that was evaluated (pop_size x dimension)
            fitness_scores: Fitness scores for each population member
        """
        # Rank-based fitness shaping (more robust than raw scores)
        fitness_array = np.array(fitness_scores)
        fitness_ranks = np.argsort(
            np.argsort(-fitness_array)
        )  # Higher rank = better fitness

        # Normalize ranks to zero mean, unit std (with numerical stability)
        rank_mean = np.mean(fitness_ranks)
        rank_std = np.std(fitness_ranks)

        if rank_std < EPS:
            # All fitness scores are identical - use uniform weights (no gradient)
            weighted_gradient = np.zeros(self.dimension, dtype=np.float32)
        else:
            # Use utility-based weights instead of zero-mean normalized ranks
            # Convert ranks to utilities (higher rank = higher utility)
            utilities = fitness_ranks.astype(np.float32)
            utility_weights = utilities - np.mean(utilities)  # Zero-mean utilities

            # Check if weights sum to approximately zero
            weights_sum = np.sum(utility_weights)
            if abs(weights_sum) < EPS:
                # If weights sum to zero, use simple ranking approach
                # Only use top half of population for gradient
                top_half_mask = utilities >= np.median(utilities)
                if np.sum(top_half_mask) == 0:
                    weighted_gradient = np.zeros(self.dimension, dtype=np.float32)
                else:
                    # Transform to logit space for gradient computation
                    eps_bound = 1e-6
                    mean_clipped = np.clip(self.mean, eps_bound, 1.0 - eps_bound)
                    mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))

                    pop_clipped = np.clip(population, eps_bound, 1.0 - eps_bound)
                    pop_logit = np.log(pop_clipped / (1.0 - pop_clipped))

                    search_directions = pop_logit - mean_logit[None, :]
                    weighted_gradient = np.mean(
                        search_directions[top_half_mask], axis=0
                    ) / (self.sigma + EPS)
            else:
                # Compute weighted gradient for mean update in logit space
                eps_bound = 1e-6
                mean_clipped = np.clip(self.mean, eps_bound, 1.0 - eps_bound)
                mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))

                pop_clipped = np.clip(population, eps_bound, 1.0 - eps_bound)
                pop_logit = np.log(pop_clipped / (1.0 - pop_clipped))

                search_directions = pop_logit - mean_logit[None, :]
                weighted_gradient = np.sum(
                    search_directions * utility_weights[:, None], axis=0
                ) / (weights_sum * (self.sigma + EPS))

        # Update mean in logit space and transform back
        eps_bound = 1e-6
        mean_clipped = np.clip(self.mean, eps_bound, 1.0 - eps_bound)
        mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))
        new_mean_logit = mean_logit + self.mean_lr * weighted_gradient
        new_mean = 1.0 / (1.0 + np.exp(-new_mean_logit))

        # Adaptive sigma update based on fitness diversity
        fitness_std = float(np.std(fitness_scores))
        target_diversity = 1e-3  # Target minimum diversity
        sigma_multiplier = math.exp(self.sigma_lr * (fitness_std - target_diversity))

        # Update sigma with bounds
        new_sigma = np.clip(
            self.sigma * sigma_multiplier, self.min_sigma, self.max_sigma
        )

        # Apply updates
        self.mean = new_mean
        self.sigma = float(new_sigma)

        # Update best candidate tracking
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_candidate = population[best_idx].copy()

        self.generation += 1

    def run(self) -> None:
        population = self._sample_population()
        if self.env is not None:
            _, fitness, _, _, _ = self.env.step(population)
        else:
            # TODO case when the fitness is not none
            fitness = "????"
        self._update_parameters(population, fitness)

        self.metrics.log_dict(
            {
                "generation": self.generation,
                "mean_fitness": float(np.mean(fitness)),
                "max_fitness": float(np.max(fitness)),
                "sigma": self.sigma,
            }
        )

    async def run_async(self) -> None:
        pass
