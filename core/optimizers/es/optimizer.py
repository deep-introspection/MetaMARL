from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

from core.optimizers.base import Optimizer

if TYPE_CHECKING:
    from core.optimizers.es.config import ESConfig


# TODO move this into constants
EPS = 1e-8


class ESOptimizer(Optimizer):
    def __init__(self, config: ESConfig):
        super().__init__(config)

        # --- hyperparameters --
        self.dimension = config.dimension
        self.mean_lr = config.mean_lr

        # TODO sigma anneal
        self.sigma_lr = config.sigma_lr
        self.min_sigma = config.min_sigma
        self.max_sigma = config.max_sigma
        self.break_symmetry = config.break_symmetry

        # --- runtime state ---
        # Initialize search distribution from initial_mean or center of unit cube
        if config.initial_mean is not None:
            self.mean = np.array(config.initial_mean, dtype=np.float32)
        else:
            self.mean = np.full(shape=config.dimension, fill_value=0.5, dtype=np.float32)
        self.sigma = float(config.sigma)

        # Random number generator
        self.rng = np.random.default_rng(config.seed)

        # History tracking
        self.generation = 0
        self.fitness_baseline = None
        self.best_fitness = -float("inf")
        self.best_candidate = self.mean.copy()
        self.best_mechanism_idx: int | None = None
        self.population_history: list[tuple[np.ndarray, np.ndarray]] = []

        # TODO move this to generalized optimizer
        self.no_improve_steps = 0
        self.convergence_patience = config.convergence_patience
        self.convergence_eps = config.convergence_eps
        self.converged_once = False

    @property
    def batch_capacity(self) -> int:
        return self._batch_capacity

    @batch_capacity.setter
    def batch_capacity(self, value: int) -> None:
        if value <= 0:
            raise ValueError("population_size must be positive")

        if not self.break_symmetry and value % 2 != 0:
            raise ValueError(
                f"Antithetic ES requires even batch size, got {value}. "
                "Either increase num_envs_per_env_runner or enable break_symmetry."
            )
        self._batch_capacity = value

    # TODO refactor this to Env_Runner sampler
    def _sample_population(self) -> np.ndarray:
        """Sample population for current generation using antithetic sampling.

        Uses logit reparametrization to avoid boundary clipping and
        antithetic sampling to reduce variance.

        Returns:
            Population matrix of shape (population_size, dimension)
        """
        # Ensure even population size for antithetic sampling
        half_pop = self._batch_capacity // 2
        remaining = self._batch_capacity - (2 * half_pop)

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

        if self.break_symmetry and self._batch_capacity % 2 == 0 and half_pop > 0:
            noise_matrix[-1] = self.rng.standard_normal(
                (self.dimension,), dtype=np.float32
            )

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
    # TODO review this and make sure faster compute
    def _update_parameters(
        self,
        population: np.ndarray,
        fitness_scores: list[float],
    ) -> None:
        eps = 1e-8
        fitness = np.asarray(fitness_scores, dtype=np.float32)

        # Fitness whitening
        f_mean = np.mean(fitness)
        f_std = np.std(fitness) + eps
        fitness = (fitness - f_mean) / f_std

        # Logit transform
        eps_bound = 1e-6
        mean_clipped = np.clip(self.mean, eps_bound, 1 - eps_bound)
        mean_logit = np.log(mean_clipped / (1 - mean_clipped))

        pop_clipped = np.clip(population, eps_bound, 1 - eps_bound)
        pop_logit = np.log(pop_clipped / (1 - pop_clipped))

        # Approximate antithetic noise
        eps_est = (pop_logit - mean_logit[None, :]) / (self.sigma + eps)

        # Mirrored gradient
        N = len(fitness)
        half = N // 2

        # Detect if strict antithetic symmetry holds
        strict_antithetic = self._batch_capacity % 2 == 0

        if self.break_symmetry and strict_antithetic and half > 0:
            f_pos = fitness[:half]
            f_neg = fitness[half : 2 * half]
            eps_pos = eps_est[:half]

            gradient = np.mean((f_pos - f_neg)[:, None] * eps_pos, axis=0)
        else:
            # Fall back to full ES estimator when symmetry is broken
            gradient = np.mean(fitness[:, None] * eps_est, axis=0)

        # Gradient clipping
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 5.0:
            gradient *= 5.0 / (grad_norm + eps)

        # Mean update
        new_mean_logit = mean_logit + self.mean_lr * gradient
        self.mean = (1.0 / (1.0 + np.exp(-new_mean_logit))).astype(np.float32)

        # Sigma update: 1/5 success rule
        success_rate = np.mean(fitness > 0)
        target = 0.2

        sigma_multiplier = math.exp(self.sigma_lr * (success_rate - target))

        self.sigma = float(
            np.clip(self.sigma * sigma_multiplier, self.min_sigma, self.max_sigma)
        )
        # Soft anchor toward mid sigma to prevent saturation
        sigma_mid = 0.5 * (self.min_sigma + self.max_sigma)
        self.sigma = 0.97 * self.sigma + 0.03 * sigma_mid

        # Track best
        best_idx = int(np.argmax(fitness_scores))
        best_fitness = float(fitness_scores[best_idx])

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_candidate = population[best_idx].copy()
            self.best_mechanism_idx = best_idx

        self.generation += 1

    def run(self) -> None:
        logger.info(
            "[ES] Generation started | gen=%d | sigma=%.5f | mean_norm=%.4f",
            self.generation,
            self.sigma,
            float(np.linalg.norm(self.mean)),
        )

        if self.env is None:
            raise RuntimeError("ESOptimizer requires a RegulatorEnv")

        population = self._sample_population()
        _, fitness, _, _, _ = self.env.step(population)
        fitness = np.asarray(fitness, dtype=np.float32)

        if not any(np.isfinite(f) for f in fitness):
            logger.error("[Regulator] No valid fitness produced for ANY mechanism")

        # TODO check why certain episodes return empty fitness
        if fitness.size == 0:
            logger.warning("[ES] No fitness returned — skipping update")
            return {"converged": False, "best_fitness": self.best_fitness}

        if not np.all(np.isfinite(fitness)):
            raise RuntimeError("Non-finite fitness detected")

        # Store population history for visualization
        self.population_history.append((population.copy(), fitness.copy()))

        best = float(fitness.max())
        var = float(fitness.var())

        improved = best > self.best_fitness + self.convergence_eps
        best_idx = int(fitness.argmax())

        if improved:
            self.best_fitness = best
            self.best_candidate = population[best_idx].copy()
            self.best_mechanism_idx = best_idx
            self.no_improve_steps = 0
        else:
            self.no_improve_steps += 1

        self._update_parameters(population, fitness)
        self.generation += 1

        converged = self.no_improve_steps >= self.convergence_patience

        if converged and not self.converged_once:
            logger.info(
                "[ES] CONVERGENCE REACHED | "
                f"gen={self.generation} | "
                f"best_fitness={self.best_fitness:.4f} | "
                f"sigma={self.sigma:.4f} | "
                f"var={var:.4f}"
            )
            self.converged_once = True

        self.metrics.log_dict(
            {
                "es/generation": self.generation,
                "es/best_fitness": self.best_fitness,
                "es/mean_fitness": float(fitness.mean()),
                "es/fitness_var": var,
                "es/sigma": self.sigma,
                "es/no_improve_steps": self.no_improve_steps,
            }
        )
        logger.info(
            "[ES] "
            f"gen={self.generation} | "
            f"best={self.best_fitness:.4f} | "
            f"mean={fitness.mean():.4f}±{fitness.std():.4f} | "
            f"var={var:.4f} | "
            f"sigma={self.sigma:.4f} | "
            f"no_improve={self.no_improve_steps}"
        )

        # Get best trajectory from env if available
        best_trajectory = None
        if hasattr(self.env, "trajectories") and self.best_mechanism_idx is not None:
            best_trajectory = self.env.trajectories.get(self.best_mechanism_idx)

        return {
            "converged": converged,
            "best_fitness": self.best_fitness,
            "best_trajectory": best_trajectory,
            "population_history": self.population_history,
        }
