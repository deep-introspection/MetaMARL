"""Algorithmic unit tests for the Evolution Strategies optimizer.

These tests exercise ``ESOptimizer`` on deterministic synthetic fitness
landscapes without Ray, W&B, or a ``RegulatorEnv``. They drive the two
primitives that ``ESOptimizer.run()`` composes -- ``_sample_population`` and
``_update_parameters`` -- so that the search dynamics (antithetic sampling in
logit space, sigma adaptation, convergence) are checked in isolation from the
distributed runtime.

The population size is set through ``batch_capacity`` and the search dimension
through ``ESConfig.dimension``; in production both are derived from the inner
optimizer and the mechanism space respectively.
"""

import numpy as np
import pytest

from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer


def make_es(
    *,
    dimension: int,
    pop_size: int,
    seed: int | None = 0,
    **training,
) -> ESOptimizer:
    """Build a standalone ``ESOptimizer`` with an explicit population size."""
    cfg = ESConfig().training(**training)
    cfg.dimension = dimension
    if seed is not None:
        cfg.debugging(seed=seed)
    opt = ESOptimizer(config=cfg)
    opt.batch_capacity = pop_size
    return opt


def run_generations(opt: ESOptimizer, env, n: int) -> None:
    """Run ``n`` ES generations against a synthetic environment.

    Mirrors the core of ``ESOptimizer.run()`` (sample -> evaluate -> update)
    without the World/reporting side effects.
    """
    for _ in range(n):
        population = opt._sample_population()
        _, fitness, *_ = env.step(population)
        opt._update_parameters(population, np.asarray(fitness, dtype=np.float32))
        opt.generation += 1


# Test ES algorithmic convergence with deterministic test environments
class QuadraticEnv:
    def __init__(self, optimum: np.ndarray):
        self.optimum = optimum

    def step(self, population: np.ndarray):
        fitness = -np.sum((population - self.optimum) ** 2, axis=1)
        return None, fitness, None, None, None


@pytest.mark.unit
def test_sample_population_shape_and_bounds():
    opt: ESOptimizer = make_es(dimension=4, pop_size=10)

    pop = opt._sample_population()

    assert pop.shape == (10, 4)
    assert np.all(pop > 0.0)
    assert np.all(pop < 1.0)


@pytest.mark.unit
def test_antithetic_sampling_symmetry():
    opt: ESOptimizer = make_es(dimension=4, pop_size=10, break_symmetry=False)

    pop = opt._sample_population()

    half = pop.shape[0] // 2
    eps = 1e-6

    def logit(x):
        clipped = np.clip(x, eps, 1 - eps)
        return np.log(clipped / (1 - clipped))

    mean_logit = logit(opt.mean)
    d1 = logit(pop[:half]) - mean_logit
    d2 = logit(pop[half : 2 * half]) - mean_logit

    assert np.allclose(d1, -d2, atol=1e-5)


@pytest.mark.unit
def test_sigma_respects_bounds():
    opt: ESOptimizer = make_es(
        dimension=2,
        pop_size=6,
        sigma=0.2,
        min_sigma=0.05,
        max_sigma=0.3,
        break_symmetry=True,
    )

    population = opt._sample_population()
    fitness = np.random.randn(population.shape[0])

    for _ in range(20):
        opt._update_parameters(population, fitness)
        assert opt.min_sigma <= opt.sigma <= opt.max_sigma


@pytest.mark.unit
def test_es_converges_on_quadratic():
    np.random.seed(0)

    optimum = np.array([0.8, 0.2, 0.6], dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=3,
        pop_size=32,
        sigma=0.25,
        mean_lr=0.05,
        sigma_lr=0.10,
        break_symmetry=True,
        seed=0,
    )
    env = QuadraticEnv(optimum)

    initial_dist = np.linalg.norm(opt.mean - optimum)

    run_generations(opt, env, 200)

    final_dist = np.linalg.norm(opt.mean - optimum)

    assert final_dist < initial_dist * 0.2


@pytest.mark.unit
def test_es_update_moves_toward_optimum():
    np.random.seed(0)

    optimum = np.array([0.9, 0.1, 0.7], dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=3,
        pop_size=40,
        sigma=0.15,
        mean_lr=0.05,
        sigma_lr=0.0,
        break_symmetry=True,
        seed=0,
    )
    env = QuadraticEnv(optimum)

    before = opt.mean.copy()
    dist_before = np.linalg.norm(before - optimum)

    run_generations(opt, env, 1)

    after = opt.mean
    dist_after = np.linalg.norm(after - optimum)

    assert dist_after < dist_before, "Gradient update moved AWAY from optimum"


@pytest.mark.unit
def test_sigma_does_not_explode_or_collapse():
    np.random.seed(1)

    optimum = np.array([0.7, 0.3, 0.6], dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=3,
        pop_size=30,
        sigma=0.25,
        mean_lr=0.03,
        sigma_lr=0.2,
        break_symmetry=True,
        seed=1,
    )
    env = QuadraticEnv(optimum)

    run_generations(opt, env, 400)

    assert not np.isnan(opt.sigma)
    assert not np.isinf(opt.sigma)
    assert opt.min_sigma <= opt.sigma <= opt.max_sigma


@pytest.mark.unit
def test_logit_respects_bounds():
    np.random.seed(0)

    optimum = np.array([0.001, 0.999, 0.5], dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=3,
        pop_size=40,
        sigma=0.3,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )
    env = QuadraticEnv(optimum)

    run_generations(opt, env, 400)

    assert np.all(opt.mean > 0.0)
    assert np.all(opt.mean < 1.0)


@pytest.mark.unit
def test_high_dimensional_convergence():
    np.random.seed(0)

    dim = 64
    optimum = np.random.uniform(0.1, 0.9, size=dim).astype(np.float32)

    opt: ESOptimizer = make_es(
        dimension=dim,
        pop_size=128,
        sigma=0.3,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )
    env = QuadraticEnv(optimum)

    initial_dist = np.linalg.norm(opt.mean - optimum)

    run_generations(opt, env, 300)

    final_dist = np.linalg.norm(opt.mean - optimum)

    assert final_dist < 0.4 * initial_dist


@pytest.mark.unit
def test_sigma_is_bounded_and_not_pathological():
    np.random.seed(1)
    optimum = np.array([0.7, 0.3, 0.6], dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=3,
        pop_size=30,
        sigma=0.25,
        mean_lr=0.03,
        sigma_lr=0.2,
        break_symmetry=True,
        seed=1,
        min_sigma=1e-3,
        max_sigma=0.5,
    )
    env = QuadraticEnv(optimum)

    sigmas = []
    for _ in range(200):
        run_generations(opt, env, 1)
        sigmas.append(opt.sigma)

    # Always bounded
    assert all(opt.min_sigma <= s <= opt.max_sigma for s in sigmas)

    # Not instantly saturating due to a bug (tweak threshold if needed)
    # e.g., sigma shouldn't be at max for >95% of steps
    frac_at_max = np.mean(np.isclose(sigmas, opt.max_sigma))
    assert frac_at_max < 0.95


@pytest.mark.unit
def test_sigma_stays_finite_and_stable():
    np.random.seed(1)

    optimum = np.array([0.7, 0.3, 0.6], dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=3,
        pop_size=30,
        sigma=0.25,
        mean_lr=0.03,
        sigma_lr=0.2,
        break_symmetry=True,
        seed=1,
        min_sigma=1e-3,
        max_sigma=0.5,
    )
    env = QuadraticEnv(optimum)

    sigmas = []
    for _ in range(300):
        run_generations(opt, env, 1)
        sigmas.append(opt.sigma)

    # Finite & bounded
    assert np.all(np.isfinite(sigmas))
    assert np.min(sigmas) >= opt.min_sigma
    assert np.max(sigmas) <= opt.max_sigma


class NoisyQuadraticEnv:
    def __init__(self, optimum, noise_std=0.1):
        self.optimum = optimum
        self.noise_std = noise_std

    def step(self, population):
        diff = population - self.optimum[None, :]
        fitness = -np.sum(diff**2, axis=1)
        noise = np.random.normal(0, self.noise_std, size=fitness.shape)
        return None, fitness + noise, None, None, None


@pytest.mark.unit
def test_es_converges_on_noisy_quadratic():
    dim = 3
    runs = 5
    improvements = []

    for seed in range(runs):
        np.random.seed(seed)

        optimum = np.array([0.8, 0.2, 0.6], dtype=np.float32)

        opt: ESOptimizer = make_es(
            dimension=dim,
            pop_size=48,
            sigma=0.35,
            mean_lr=0.05,
            sigma_lr=0.15,
            break_symmetry=True,
            seed=seed,
        )
        env = NoisyQuadraticEnv(optimum, noise_std=0.1)

        initial_dist = np.linalg.norm(opt.mean - optimum)

        run_generations(opt, env, 300)

        final_dist = np.linalg.norm(opt.mean - optimum)
        improvements.append(final_dist / initial_dist)

    # On average, should reduce error by at least 25%
    assert np.mean(improvements) < 0.70


class MultimodalEnv:
    def __init__(self, dim):
        self.dim = dim

    def step(self, population):
        # Two optima: near 0.2 and near 0.8
        peak1 = -np.sum((population - 0.2) ** 2, axis=1)
        peak2 = -np.sum((population - 0.8) ** 2, axis=1)
        fitness = np.maximum(peak1, peak2)
        return None, fitness, None, None, None


@pytest.mark.unit
def test_es_finds_a_global_peak_in_multimodal():
    np.random.seed(0)

    dim = 3
    opt: ESOptimizer = make_es(
        dimension=dim,
        pop_size=64,
        sigma=0.4,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )
    env = MultimodalEnv(dim)

    run_generations(opt, env, 400)

    # Should converge near either peak
    d1 = np.linalg.norm(opt.mean - 0.2)
    d2 = np.linalg.norm(opt.mean - 0.8)
    assert min(d1, d2) < 0.05


class StepRidgeEnv:
    def step(self, pop):
        return None, np.where(pop.mean(axis=1) > 0.6, 1.0, 0.0), None, None, None


@pytest.mark.unit
def test_es_handles_sparse_threshold_rewards():
    np.random.seed(0)

    dim = 4
    opt: ESOptimizer = make_es(
        dimension=dim,
        pop_size=64,
        sigma=0.4,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )
    env = StepRidgeEnv()

    run_generations(opt, env, 400)

    # Should push population mean above threshold
    assert opt.mean.mean() > 0.6


class DelayedEnv:
    def step(self, pop):
        base = -np.sum((pop - 0.7) ** 2, axis=1)
        delay_noise = np.random.randn(len(pop)) * 0.2
        return None, base + delay_noise, None, None, None


@pytest.mark.unit
def test_es_converges_under_delayed_reward():
    np.random.seed(0)

    dim = 5
    optimum = np.full(dim, 0.7, dtype=np.float32)

    opt: ESOptimizer = make_es(
        dimension=dim,
        pop_size=64,
        sigma=0.35,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )
    env = DelayedEnv()

    initial_dist = np.linalg.norm(opt.mean - optimum)

    run_generations(opt, env, 400)

    final_dist = np.linalg.norm(opt.mean - optimum)

    # Improvement under delay
    assert final_dist < 0.6 * initial_dist


class PPOProxyEnv:
    def __init__(self, dim, optimum):
        self.dim = dim
        self.optimum = optimum

    def step(self, population):
        rewards = []

        for x in population:
            perf = 0
            for _ in range(20):  # simulate PPO epochs
                perf += -np.linalg.norm(x - self.optimum) + np.random.randn() * 0.05
            rewards.append(perf)

        return None, np.array(rewards), None, None, None


@pytest.mark.unit
def test_es_converges_on_ppo_proxy():
    np.random.seed(0)

    dim = 6
    optimum = np.random.uniform(0.2, 0.8, size=dim).astype(np.float32)

    opt: ESOptimizer = make_es(
        dimension=dim,
        pop_size=80,
        sigma=0.4,
        mean_lr=0.04,
        sigma_lr=0.12,
        seed=0,
    )
    env = PPOProxyEnv(dim, optimum)

    initial_dist = np.linalg.norm(opt.mean - optimum)

    run_generations(opt, env, 600)

    final_dist = np.linalg.norm(opt.mean - optimum)

    # PPO-style learning is very noisy → modest threshold
    assert final_dist < 0.7 * initial_dist


class MultimodalPPOEnv:
    def __init__(self, dim):
        self.dim = dim
        self.opt1 = np.full(dim, 0.25)
        self.opt2 = np.full(dim, 0.75)

    def step(self, population):
        rewards = []
        for x in population:
            perf1 = -np.linalg.norm(x - self.opt1)
            perf2 = -np.linalg.norm(x - self.opt2)
            perf = max(perf1, perf2)

            # PPO noise simulation
            for _ in range(15):
                perf += np.random.randn() * 0.05

            rewards.append(perf)

        return None, np.array(rewards), None, None, None


@pytest.mark.unit
def test_es_finds_global_optimum_under_ppo_noise():
    np.random.seed(0)

    dim = 4
    opt: ESOptimizer = make_es(
        dimension=dim,
        pop_size=96,
        sigma=0.45,
        mean_lr=0.04,
        sigma_lr=0.12,
        break_symmetry=True,
        seed=0,
    )
    env = MultimodalPPOEnv(dim)

    run_generations(opt, env, 800)

    d1 = np.linalg.norm(opt.mean - 0.25)
    d2 = np.linalg.norm(opt.mean - 0.75)

    assert min(d1, d2) < 0.2
