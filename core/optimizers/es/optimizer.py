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
    """Outer-loop Evolution Strategies optimizer for mechanism search.

    Implements a separable NES-style update in logit space with antithetic
    (mirrored) sampling and a 1/5-success step-size rule.  The search
    distribution is an isotropic Gaussian
    :math:`\\mathcal{N}(\\mu, \\sigma^2 I)` maintained in the logit-transformed
    space of the ``[0, 1]``-bounded mechanism parameter vector.

    **Sampling (antithetic).**  Each generation draws :math:`N/2` noise
    vectors :math:`\\varepsilon_i \\sim \\mathcal{N}(0, I)` and mirrors them to
    form :math:`N` candidates:

    .. math::

        z_i^{\\pm} = \\sigma^{-1}(\\mu) \\pm \\sigma \\, \\varepsilon_i

        x_i^{\\pm} = \\text{sigmoid}(z_i^{\\pm})

    where :math:`\\sigma^{-1}` is the logit function and :math:`\\text{sigmoid}`
    maps back to ``[0, 1]``.

    **Mean update.**  Fitness scores are whitened and the mean is updated via
    the natural gradient in logit space:

    .. math::

        g = \\frac{1}{N} \\sum_i \\hat{f}_i \\, \\tilde{\\varepsilon}_i, \\qquad
        \\mu_{\\text{logit}} \\leftarrow \\mu_{\\text{logit}} + \\alpha_\\mu \\, g

    **Step-size adaptation (1/5-success rule).**  After each generation the
    step size is scaled according to the empirical success rate
    :math:`p_s = \\mathbb{E}[\\hat{f}_i > 0]`:

    .. math::

        \\sigma \\leftarrow \\text{clip}\\!\\left(
            \\sigma \\cdot e^{\\alpha_\\sigma (p_s - 0.2)},\\,
            \\sigma_{\\min},\\, \\sigma_{\\max}
        \\right)

    A soft anchor toward the midpoint
    :math:`\\bar\\sigma = (\\sigma_{\\min} + \\sigma_{\\max})/2` is applied to
    prevent saturation: :math:`\\sigma \\leftarrow 0.97\\sigma + 0.03\\bar\\sigma`.

    Convergence is declared when no improvement larger than
    ``convergence_eps`` is observed for ``convergence_patience`` consecutive
    generations.

    References
    ----------
    Wierstra, D. et al. (2014) "Natural Evolution Strategies"
    *Journal of Machine Learning Research*, 15, pp. 949-980.

    Rechenberg, I. (1973) *Evolutionsstrategie*.  Frommann-Holzboog.

    Attributes
    ----------
    dimension : int
        Dimensionality of the mechanism parameter vector.
    mean_lr : float
        Learning rate :math:`\\alpha_\\mu` for the mean update.
    sigma_lr : float
        Learning rate :math:`\\alpha_\\sigma` for step-size adaptation.
    min_sigma : float
        Lower bound on the step size.
    max_sigma : float
        Upper bound on the step size.
    break_symmetry : bool
        Whether to break strict antithetic symmetry on the last sample.
    mean : numpy.ndarray, shape (dimension,)
        Current mean of the search distribution in ``[0, 1]^d`` (probability
        space, not logit space).
    sigma : float
        Current step size (standard deviation in logit space).
    rng : numpy.random.Generator
        Seeded random number generator.
    generation : int
        Number of completed update steps.
    best_fitness : float
        Best fitness score observed so far.
    best_candidate : numpy.ndarray, shape (dimension,)
        Parameter vector of the best mechanism found so far.
    best_mechanism_idx : int or None
        Index of the best candidate within its generation's population array.
    population_history : list[tuple[ndarray, ndarray]]
        Sequence of ``(population, fitness)`` arrays for each generation.
    no_improve_steps : int
        Number of consecutive generations without ``convergence_eps``
        improvement.
    convergence_patience : int
        Patience threshold before convergence is declared.
    convergence_eps : float
        Minimum improvement to reset the patience counter.
    converged_once : bool
        Latching flag set to ``True`` the first time convergence is detected.
    """

    def __init__(self, config: ESConfig):
        """Initialise the ES optimizer.

        Parameters
        ----------
        config : ESConfig
            Fully specified ES configuration object.
        """
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
            self.mean = np.full(
                shape=config.dimension, fill_value=0.5, dtype=np.float32
            )
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
        """Population size for the current generation.

        Returns
        -------
        int
            The number of candidate mechanisms sampled each generation.
            Set via the ``batch_capacity`` setter, which is called by
            :class:`~core.optimizers.bilevel.BilevelConfig` to synchronise
            the ES population size with the inner RL optimizer's rollout batch.
        """
        return self._batch_capacity

    @batch_capacity.setter
    def batch_capacity(self, value: int) -> None:
        """Set the population size, enforcing antithetic sampling constraints.

        Parameters
        ----------
        value : int
            Desired population size.  Must be positive.  When antithetic
            sampling is active (``break_symmetry=False``), must also be even
            so that mirrored pairs can be formed.

        Raises
        ------
        ValueError
            If ``value <= 0``, or if ``value`` is odd and
            ``break_symmetry=False``.
        """
        if value <= 0:
            raise ValueError("population_size must be positive")

        if value == 1:
            self._batch_capacity = value
            return

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
        """Update the search distribution parameters given a scored population.

        Applies the NES mean update and the 1/5-success step-size rule.  All
        gradient computations are performed in logit space to respect the
        ``[0, 1]`` parameter bounds.

        **Mean update:**

        .. math::

            g = \\frac{1}{N} \\sum_i \\hat{f}_i \\, \\tilde{\\varepsilon}_i, \\qquad
            \\mu_{\\text{logit}} \\leftarrow \\mu_{\\text{logit}} + \\alpha_\\mu \\, g

        with gradient clipping at :math:`\\|g\\| \\le 5`.

        **Step-size update (1/5-success rule):**

        .. math::

            \\sigma \\leftarrow \\text{clip}\\!\\left(
                \\sigma \\cdot e^{\\alpha_\\sigma (p_s - 0.2)},\\,
                \\sigma_{\\min},\\, \\sigma_{\\max}
            \\right)

        followed by a soft anchor:
        :math:`\\sigma \\leftarrow 0.97\\sigma + 0.03\\bar\\sigma`.

        Parameters
        ----------
        population : numpy.ndarray, shape (N, d)
            Population of mechanism candidates in ``[0, 1]^d`` sampled by
            :meth:`_sample_population`.
        fitness_scores : list[float]
            Scalar fitness values for each candidate, indexed consistently with
            the rows of ``population``.
        """
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

    def run(self) -> dict:
        """Execute one generation of the ES outer loop.

        Samples a population of mechanism candidates, evaluates them through
        the inner RL optimizer via the regulator environment, updates the search
        distribution, and checks for convergence.

        Returns
        -------
        dict
            A per-generation summary dictionary with the following keys:

            ``"converged"`` : bool
                ``True`` if the no-improvement counter has exceeded
                ``convergence_patience``.
            ``"best_fitness"`` : float
                Best fitness observed across all generations (not just this one).
            ``"best_trajectory"`` : list[dict] or None
                Environment trajectory corresponding to the best mechanism, if
                available from ``self.env.trajectories``; otherwise ``None``.
            ``"population_history"`` : list[tuple[ndarray, ndarray]]
                Full population/fitness history accumulated so far.

        Raises
        ------
        RuntimeError
            If ``self.env`` is ``None`` (no regulator environment attached) or
            if non-finite fitness values are detected after evaluation.
        """
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
