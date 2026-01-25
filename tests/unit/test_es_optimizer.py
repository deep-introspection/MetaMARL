import asyncio

import numpy as np
import pytest

from src.es.config import ESConfig
from src.es.optimizer import ESOptimizer

# Test ES algorithmic convergence with deterministic test environment


class QuadraticEnv:
    def __init__(self, optimum: np.ndarray):
        self.optimum = optimum

    async def step(self, population: np.ndarray):
        fitness = -np.sum((population - self.optimum) ** 2, axis=1)
        return None, fitness, None, None, None

    # def step(self, population: np.ndarray):
    #     fitness = -np.sum((population - self.optimum) ** 2, axis=1)
    #     return None, fitness, None, None, None


@pytest.mark.unit
def test_sample_population_shape_and_bounds():
    cfg = ESConfig().training(dimension=4, pop_size=10)
    opt: ESOptimizer = cfg.build_optimizer()

    pop = opt._sample_population()

    assert pop.shape == (10, 4)
    assert np.all(pop > 0.0)
    assert np.all(pop < 1.0)


@pytest.mark.unit
def test_antithetic_sampling_symmetry():
    cfg = ESConfig().training(dimension=4, pop_size=10, break_symmetry=False)
    opt: ESOptimizer = cfg.build_optimizer()

    pop = opt._sample_population()

    half = pop.shape[0] // 2
    eps = 1e-6

    logit = lambda x: np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps)))

    mean_logit = logit(opt.mean)
    d1 = logit(pop[:half]) - mean_logit
    d2 = logit(pop[half : 2 * half]) - mean_logit

    assert np.allclose(d1, -d2, atol=1e-5)


@pytest.mark.unit
def test_sigma_respects_bounds():
    cfg = ESConfig().training(
        dimension=2,
        pop_size=6,
        sigma=0.2,
        min_sigma=0.05,
        max_sigma=0.3,
        break_symmetry=True,
    )
    opt: ESOptimizer = cfg.build_optimizer()

    population = opt._sample_population()
    fitness = np.random.randn(cfg.pop_size)

    for _ in range(20):
        opt._update_parameters(population, fitness)
        assert cfg.min_sigma <= opt.sigma <= cfg.max_sigma


@pytest.mark.unit
def test_es_converges_on_quadratic():
    np.random.seed(0)

    optimum = np.array([0.8, 0.2, 0.6], dtype=np.float32)

    cfg = ESConfig().training(
        dimension=3,
        pop_size=32,
        sigma=0.25,
        mean_lr=0.05,
        sigma_lr=0.10,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    initial_dist = np.linalg.norm(opt.mean - optimum)

    async def run_steps(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run_steps(200))

    final_dist = np.linalg.norm(opt.mean - optimum)

    assert final_dist < initial_dist * 0.2


@pytest.mark.unit
def test_es_update_moves_toward_optimum():
    np.random.seed(0)

    optimum = np.array([0.9, 0.1, 0.7], dtype=np.float32)

    cfg = ESConfig().training(
        dimension=3,
        pop_size=40,
        sigma=0.15,
        mean_lr=0.05,
        sigma_lr=0.0,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    before = opt.mean.copy()
    dist_before = np.linalg.norm(before - optimum)

    asyncio.run(opt.run_async())

    after = opt.mean
    dist_after = np.linalg.norm(after - optimum)

    assert dist_after < dist_before, "Gradient update moved AWAY from optimum"


@pytest.mark.unit
def test_sigma_does_not_explode_or_collapse():
    np.random.seed(1)

    optimum = np.array([0.7, 0.3, 0.6], dtype=np.float32)

    cfg = ESConfig().training(
        dimension=3,
        pop_size=30,
        sigma=0.25,
        mean_lr=0.03,
        sigma_lr=0.2,
        break_symmetry=True,
        seed=1,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(400))

    assert not np.isnan(opt.sigma)
    assert not np.isinf(opt.sigma)
    assert opt.min_sigma <= opt.sigma <= opt.max_sigma


@pytest.mark.unit
def test_logit_respects_bounds():
    np.random.seed(0)

    optimum = np.array([0.001, 0.999, 0.5], dtype=np.float32)

    cfg = ESConfig().training(
        dimension=3,
        pop_size=40,
        sigma=0.3,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(400))

    assert np.all(opt.mean > 0.0)
    assert np.all(opt.mean < 1.0)


@pytest.mark.unit
def test_high_dimensional_convergence():
    np.random.seed(0)

    dim = 64
    optimum = np.random.uniform(0.1, 0.9, size=dim).astype(np.float32)

    cfg = ESConfig().training(
        dimension=dim,
        pop_size=128,
        sigma=0.3,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    initial_dist = np.linalg.norm(opt.mean - optimum)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(300))

    final_dist = np.linalg.norm(opt.mean - optimum)

    assert final_dist < 0.4 * initial_dist


@pytest.mark.unit
def test_sigma_is_bounded_and_not_pathological():
    np.random.seed(1)
    optimum = np.array([0.7, 0.3, 0.6], dtype=np.float32)

    cfg = ESConfig().training(
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

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    sigmas = []

    async def run(n):
        for _ in range(n):
            await opt.run_async()
            sigmas.append(opt.sigma)

    asyncio.run(run(200))

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

    cfg = ESConfig().training(
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

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = QuadraticEnv(optimum)

    sigmas = []

    async def run(n):
        for _ in range(n):
            await opt.run_async()
            sigmas.append(opt.sigma)

    asyncio.run(run(300))

    # Finite & bounded
    assert np.all(np.isfinite(sigmas))
    assert np.min(sigmas) >= opt.min_sigma
    assert np.max(sigmas) <= opt.max_sigma


class NoisyQuadraticEnv:
    def __init__(self, optimum, noise_std=0.1):
        self.optimum = optimum
        self.noise_std = noise_std

    async def step(self, population):
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

        cfg = ESConfig().training(
            dimension=dim,
            pop_size=48,
            sigma=0.35,
            mean_lr=0.05,
            sigma_lr=0.15,
            break_symmetry=True,
            seed=seed,
        )

        opt: ESOptimizer = cfg.build_optimizer()
        opt.env = NoisyQuadraticEnv(optimum, noise_std=0.1)

        initial_dist = np.linalg.norm(opt.mean - optimum)

        async def run(n):
            for _ in range(n):
                await opt.run_async()

        asyncio.run(run(300))

        final_dist = np.linalg.norm(opt.mean - optimum)
        improvements.append(final_dist / initial_dist)

    # On average, should reduce error by at least 25%
    assert np.mean(improvements) < 0.70


class MultimodalEnv:
    def __init__(self, dim):
        self.dim = dim

    async def step(self, population):
        # Two optima: near 0.2 and near 0.8
        peak1 = -np.sum((population - 0.2) ** 2, axis=1)
        peak2 = -np.sum((population - 0.8) ** 2, axis=1)
        fitness = np.maximum(peak1, peak2)
        return None, fitness, None, None, None


@pytest.mark.unit
def test_es_finds_a_global_peak_in_multimodal():
    np.random.seed(0)

    dim = 3
    cfg = ESConfig().training(
        dimension=dim,
        pop_size=64,
        sigma=0.4,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = MultimodalEnv(dim)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(400))

    # Should converge near either peak
    d1 = np.linalg.norm(opt.mean - 0.2)
    d2 = np.linalg.norm(opt.mean - 0.8)
    assert min(d1, d2) < 0.05


class StepRidgeEnv:
    async def step(self, pop):
        return None, np.where(pop.mean(axis=1) > 0.6, 1.0, 0.0), None, None, None


@pytest.mark.unit
def test_es_handles_sparse_threshold_rewards():
    np.random.seed(0)

    dim = 4
    cfg = ESConfig().training(
        dimension=dim,
        pop_size=64,
        sigma=0.4,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = StepRidgeEnv()

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(400))

    # Should push population mean above threshold
    assert opt.mean.mean() > 0.6


class DelayedEnv:
    async def step(self, pop):
        base = -np.sum((pop - 0.7) ** 2, axis=1)
        delay_noise = np.random.randn(len(pop)) * 0.2
        return None, base + delay_noise, None, None, None


@pytest.mark.unit
def test_es_converges_under_delayed_reward():
    np.random.seed(0)

    dim = 5
    optimum = np.full(dim, 0.7, dtype=np.float32)

    cfg = ESConfig().training(
        dimension=dim,
        pop_size=64,
        sigma=0.35,
        mean_lr=0.05,
        sigma_lr=0.15,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = DelayedEnv()

    initial_dist = np.linalg.norm(opt.mean - optimum)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(400))

    final_dist = np.linalg.norm(opt.mean - optimum)

    # Improvement under delay
    assert final_dist < 0.6 * initial_dist


class SimulatedPPOEnv:
    def __init__(self):
        self.optimum = np.array([...])

    async def step(self, population):
        steps = 20
        rewards = []

        for theta in population:
            perf = 0
            for _ in range(steps):
                perf += -np.linalg.norm(theta - self.optimum) + np.random.randn() * 0.05
            rewards.append(perf)

        return None, np.array(rewards), None, None, None


class PPOProxyEnv:
    def __init__(self, dim, optimum):
        self.dim = dim
        self.optimum = optimum

    async def step(self, population):
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

    cfg = ESConfig().training(
        dimension=dim,
        pop_size=80,
        sigma=0.4,
        mean_lr=0.04,
        sigma_lr=0.12,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = PPOProxyEnv(dim, optimum)

    initial_dist = np.linalg.norm(opt.mean - optimum)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(600))

    final_dist = np.linalg.norm(opt.mean - optimum)

    # PPO-style learning is very noisy → modest threshold
    assert final_dist < 0.7 * initial_dist


class MultimodalPPOEnv:
    def __init__(self, dim):
        self.dim = dim
        self.opt1 = np.full(dim, 0.25)
        self.opt2 = np.full(dim, 0.75)

    async def step(self, population):
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
    cfg = ESConfig().training(
        dimension=dim,
        pop_size=96,
        sigma=0.45,
        mean_lr=0.04,
        sigma_lr=0.12,
        break_symmetry=True,
        seed=0,
    )

    opt: ESOptimizer = cfg.build_optimizer()
    opt.env = MultimodalPPOEnv(dim)

    async def run(n):
        for _ in range(n):
            await opt.run_async()

    asyncio.run(run(800))

    d1 = np.linalg.norm(opt.mean - 0.25)
    d2 = np.linalg.norm(opt.mean - 0.75)

    assert min(d1, d2) < 0.2
