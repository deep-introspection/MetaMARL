"""Evolution Strategies implementation for mechanism optimization.

This module implements the outer-loop optimization using Evolution Strategies
to search for optimal regulatory mechanism parameters in the unit hypercube.
"""

import math
from typing import List, Tuple

import numpy as np

from examples.config import EPS, ES_CONFIG


class EvolutionStrategies:
    """Evolution Strategies optimizer for mechanism parameter search.

    This class implements a simple Gaussian Evolution Strategies algorithm
    with rank-based fitness shaping for robust optimization.
    """

    def __init__(
        self,
        dimension: int = 5,
        population_size: int = 8,
        sigma: float = 0.15,
        mean_learning_rate: float = 0.1,
        sigma_learning_rate: float = 0.05,
        min_sigma: float = 1e-3,
        max_sigma: float = 0.5,
        random_seed: int = 0,
    ):
        """Initialize Evolution Strategies optimizer.

        Args:
            dimension: Dimensionality of search space
            population_size: Number of candidates per generation
            sigma: Initial noise standard deviation
            mean_learning_rate: Learning rate for mean updates
            sigma_learning_rate: Learning rate for sigma adaptation
            min_sigma: Minimum allowed sigma value
            max_sigma: Maximum allowed sigma value
            random_seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.population_size = population_size
        self.mean_lr = mean_learning_rate
        self.sigma_lr = sigma_learning_rate
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        # Initialize search distribution at center of unit cube
        self.mean = np.full(dimension, 0.5, dtype=np.float32)
        self.sigma = float(sigma)

        # Random number generator
        self.rng = np.random.default_rng(random_seed)

        # History tracking
        self.generation = 0
        self.best_fitness = -float("inf")
        self.best_candidate = self.mean.copy()

    def sample_population(self) -> np.ndarray:
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

    def update_parameters(
        self, population: np.ndarray, fitness_scores: List[float]
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

    def get_state(self) -> dict:
        """Get current optimizer state.

        Returns:
            Dictionary containing current state information
        """
        return {
            "generation": self.generation,
            "mean": self.mean.copy(),
            "sigma": self.sigma,
            "best_fitness": self.best_fitness,
            "best_candidate": self.best_candidate.copy(),
        }

    def set_state(self, state: dict) -> None:
        """Set optimizer state from dictionary.

        Args:
            state: State dictionary from get_state()
        """
        self.generation = state["generation"]
        self.mean = state["mean"].copy()
        self.sigma = state["sigma"]
        self.best_fitness = state["best_fitness"]
        self.best_candidate = state["best_candidate"].copy()


def sample_es_population(
    mean: np.ndarray,
    sigma: float,
    population_size: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Sample population for Evolution Strategies with antithetic sampling (standalone function).

    Uses logit reparametrization to avoid boundary clipping and
    antithetic sampling to reduce variance.

    Args:
        mean: Current mean of the search distribution
        sigma: Current standard deviation (noise level)
        population_size: Number of candidates to sample
        random_generator: NumPy random generator

    Returns:
        Population matrix of shape (population_size, dimension)
    """
    dimension = mean.size

    # Ensure even population size for antithetic sampling
    half_pop = population_size // 2
    remaining = population_size - (2 * half_pop)

    # Sample noise for half the population
    noise_half = random_generator.standard_normal(
        (half_pop, dimension), dtype=np.float32
    )

    # Create antithetic pairs (mirrored noise)
    if half_pop > 0:
        noise_matrix = np.vstack([noise_half, -noise_half])
    else:
        noise_matrix = np.empty((0, dimension), dtype=np.float32)

    # Add remaining samples if population size is odd
    if remaining > 0:
        extra_noise = random_generator.standard_normal(
            (remaining, dimension), dtype=np.float32
        )
        noise_matrix = np.vstack([noise_matrix, extra_noise])

    # Transform mean to logit space for unbounded optimization
    # logit(p) = log(p/(1-p)), inverse_logit(x) = 1/(1+exp(-x))
    eps_bound = 1e-6
    mean_clipped = np.clip(mean, eps_bound, 1.0 - eps_bound)
    mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))

    # Add noise in logit space
    population_logit = mean_logit[None, :] + sigma * noise_matrix

    # Transform back to probability space using sigmoid
    population = 1.0 / (1.0 + np.exp(-population_logit))

    return population.astype(np.float32)


def update_es_parameters(
    current_mean: np.ndarray,
    current_sigma: float,
    population: np.ndarray,
    fitness_scores: List[float],
    mean_learning_rate: float,
    sigma_learning_rate: float,
) -> Tuple[np.ndarray, float]:
    """Update ES distribution parameters using fitness-weighted gradients (standalone function).

    Uses rank-based fitness shaping for robustness to outliers.
    Works in logit space for unconstrained optimization.

    Args:
        current_mean: Current mean of search distribution
        current_sigma: Current standard deviation
        population: Population that was evaluated (pop_size x dimension)
        fitness_scores: Fitness scores for each population member
        mean_learning_rate: Learning rate for mean update
        sigma_learning_rate: Learning rate for sigma adaptation

    Returns:
        Tuple of (new_mean, new_sigma)
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
        weighted_gradient = np.zeros(current_mean.size, dtype=np.float32)
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
                weighted_gradient = np.zeros(current_mean.size, dtype=np.float32)
            else:
                # Transform to logit space for gradient computation
                eps_bound = 1e-6
                mean_clipped = np.clip(current_mean, eps_bound, 1.0 - eps_bound)
                mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))

                pop_clipped = np.clip(population, eps_bound, 1.0 - eps_bound)
                pop_logit = np.log(pop_clipped / (1.0 - pop_clipped))

                search_directions = pop_logit - mean_logit[None, :]
                weighted_gradient = np.mean(
                    search_directions[top_half_mask], axis=0
                ) / (current_sigma + EPS)
        else:
            # Compute weighted gradient for mean update in logit space
            eps_bound = 1e-6
            mean_clipped = np.clip(current_mean, eps_bound, 1.0 - eps_bound)
            mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))

            pop_clipped = np.clip(population, eps_bound, 1.0 - eps_bound)
            pop_logit = np.log(pop_clipped / (1.0 - pop_clipped))

            search_directions = pop_logit - mean_logit[None, :]
            weighted_gradient = np.sum(
                search_directions * utility_weights[:, None], axis=0
            ) / (weights_sum * (current_sigma + EPS))

    # Update mean in logit space and transform back
    eps_bound = 1e-6
    mean_clipped = np.clip(current_mean, eps_bound, 1.0 - eps_bound)
    mean_logit = np.log(mean_clipped / (1.0 - mean_clipped))
    new_mean_logit = mean_logit + mean_learning_rate * weighted_gradient
    new_mean = 1.0 / (1.0 + np.exp(-new_mean_logit))

    # Adaptive sigma update based on fitness diversity
    fitness_std = float(np.std(fitness_scores))
    target_diversity = 1e-3  # Target minimum diversity
    sigma_multiplier = math.exp(sigma_learning_rate * (fitness_std - target_diversity))

    # Update sigma with bounds
    new_sigma = np.clip(
        current_sigma * sigma_multiplier, ES_CONFIG["min_sigma"], ES_CONFIG["max_sigma"]
    )

    return new_mean, float(new_sigma)
