"""Edge cases and lifecycle of ``ESOptimizer``: validation, fixed mode, (1+1)-ES, ``run()``."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer


def make_cfg(dimension=2, **training):
    cfg = ESConfig().training(**training)
    cfg.dimension = dimension
    cfg.debugging(seed=0)
    return cfg


def make_es(dimension=2, pop_size=4, **training) -> ESOptimizer:
    opt = ESOptimizer(config=make_cfg(dimension, **training))
    opt.batch_capacity = pop_size
    return opt


def space(dimension: int):
    """Minimal mechanism-space stand-in: optimized parameter names + default vector."""
    names = [f"p{i}" for i in range(dimension)]
    default = SimpleNamespace(
        param_names=lambda: names or ["a", "b"],
        to_vector=lambda: (
            np.array([0.3, 0.7], dtype=np.float32)
            if not names
            else np.full(dimension, 0.5, dtype=np.float32)
        ),
    )
    return SimpleNamespace(optimize_params=names, default=lambda: default)


@pytest.mark.unit
class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"mean_lr": 0.0}, "mean_lr"),
            ({"sigma_lr": -0.1}, "sigma_lr"),
            ({"sigma_decay": 0.0}, "sigma_decay"),
            ({"sigma_decay": 1.5}, "sigma_decay"),
            ({"min_sigma": 0.0}, "min_sigma"),
            ({"min_sigma": 0.5, "max_sigma": 0.1}, "max_sigma"),
        ],
    )
    def test_hyperparameter_errors(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ESOptimizer(config=make_cfg(**kwargs))

    def test_negative_dimension(self):
        with pytest.raises(ValueError, match="dimension"):
            ESOptimizer(config=make_cfg(dimension=-1))

    def test_initial_mean_checks(self):
        with pytest.raises(ValueError, match="shape"):
            ESOptimizer(config=make_cfg(2, initial_mean=[0.5]))
        with pytest.raises(ValueError, match="finite"):
            ESOptimizer(config=make_cfg(2, initial_mean=[0.5, float("nan")]))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ESOptimizer(config=make_cfg(2, initial_mean=[0.5, 1.5]))
        opt = ESOptimizer(config=make_cfg(2, initial_mean=[0.2, 0.8]))
        np.testing.assert_allclose(opt.mean, [0.2, 0.8])

    def test_sigma_is_clipped_to_bounds(self):
        opt = ESOptimizer(config=make_cfg(2, sigma=5.0, max_sigma=0.4))
        assert opt.sigma == 0.4


@pytest.mark.unit
class TestBatchCapacity:
    def test_non_positive(self):
        with pytest.raises(ValueError):
            make_es(pop_size=0)

    def test_odd_requires_break_symmetry(self):
        with pytest.raises(ValueError, match="even"):
            make_es(pop_size=3)
        assert make_es(pop_size=3, break_symmetry=True).batch_capacity == 3

    def test_single_candidate_mode(self):
        opt = make_es(pop_size=1)
        assert opt.batch_capacity == 1
        np.testing.assert_allclose(opt._sample_population(), opt.mean[None, :])

    def test_fixed_mode_accepts_any_size(self):
        opt = make_es(dimension=0, pop_size=3)
        assert opt.fixed_mode
        assert opt._sample_population().shape == (3, 0)


@pytest.mark.unit
class TestFixedMode:
    def test_update_tracks_best_without_parameters(self):
        opt = make_es(dimension=0, pop_size=3)
        opt._update_parameters(np.empty((3, 0)), [1.0, 3.0, 2.0])
        assert opt.best_fitness == 3.0
        assert opt.best_mechanism_idx == 1
        assert opt.best_candidate.shape == (0,)
        assert opt.previous_population_mean_fitness == pytest.approx(2.0)


@pytest.mark.unit
class TestSingleCandidate:
    def test_one_plus_one_es_accept_reject(self):
        opt = make_es(dimension=2, pop_size=1, sigma=0.1, sigma_lr=0.5, sigma_decay=0.5)
        first = opt._sample_population()
        opt._update_parameters(first, [1.0])  # initializes the parent
        assert opt.fitness_baseline == 1.0
        np.testing.assert_allclose(opt.mean, first[0])

        sigma0 = opt.sigma
        worse = np.array([[0.1, 0.1]], dtype=np.float32)
        opt._update_parameters(worse, [0.5])  # rejected
        np.testing.assert_allclose(opt.mean, first[0])
        assert opt.sigma < sigma0  # contracts on rejection

        better = np.array([[0.9, 0.9]], dtype=np.float32)
        opt._update_parameters(better, [2.0])  # accepted
        np.testing.assert_allclose(opt.mean, [0.9, 0.9])
        assert opt.best_fitness == 2.0 and opt.fitness_baseline == 2.0

    def test_non_finite_single_fitness(self):
        opt = make_es(dimension=2, pop_size=1)
        with pytest.raises(ValueError):
            opt._update_parameters(opt._sample_population(), [float("inf")])


@pytest.mark.unit
class TestUpdateParametersValidation:
    def test_shape_errors(self):
        opt = make_es(dimension=2, pop_size=4)
        with pytest.raises(ValueError, match="2D"):
            opt._update_parameters(np.zeros(4), [1, 2, 3, 4])
        with pytest.raises(ValueError, match="do not match"):
            opt._update_parameters(np.zeros((4, 2)), [1, 2, 3])
        with pytest.raises(ValueError, match="finite"):
            opt._update_parameters(np.full((4, 2), 0.5), [1, 2, np.nan, 4])
        with pytest.raises(ValueError):
            opt._update_parameters(np.zeros((0, 2)), [])

    def test_flat_fitness_leaves_mean_unchanged(self):
        opt = make_es(dimension=2, pop_size=4, sigma_lr=0.0)
        before = opt.mean.copy()
        opt._update_parameters(opt._sample_population(), [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(opt.mean, before, atol=1e-6)

    def test_strict_antithetic_gradient_moves_toward_better_half(self):
        opt = make_es(dimension=2, pop_size=4, sigma=0.2, mean_lr=0.5, sigma_lr=0.0)
        pop = opt._sample_population()
        target = np.array([0.9, 0.9])
        fitness = -np.sum((pop - target) ** 2, axis=1)
        before = np.linalg.norm(opt.mean - target)
        opt._update_parameters(pop, fitness)
        assert np.linalg.norm(opt.mean - target) < before


@pytest.mark.unit
class TestSigmaAdaptation:
    def test_contract_expand_hold(self):
        opt = make_es(
            dimension=2,
            pop_size=4,
            sigma=0.2,
            sigma_lr=1.0,
            sigma_decay=0.5,
            min_sigma=0.01,
            max_sigma=0.5,
        )
        opt.previous_population_mean_fitness = 1.0
        assert opt._update_sigma(2.0) == "contracted" and opt.sigma == pytest.approx(
            0.1
        )
        assert opt._update_sigma(0.5) == "expanded" and opt.sigma == pytest.approx(0.2)
        assert opt._update_sigma(0.5) == "held" and opt.sigma == pytest.approx(0.2)

    def test_first_generation_only_initializes(self):
        opt = make_es(dimension=2, pop_size=4, sigma=0.2, sigma_lr=1.0)
        assert opt._update_sigma(1.0) == "initialized"
        assert opt.sigma == pytest.approx(0.2)
        assert opt.previous_population_mean_fitness == 1.0


class _FakeRegulatorEnv:
    def __init__(self, dimension, fitness_fn):
        # this branch reads the optimized parameter names from env.m_space
        names = [f"p{i}" for i in range(dimension)]
        default = SimpleNamespace(
            param_names=lambda: names or ["a", "b"],
            to_vector=lambda: (
                np.full(dimension, 0.5, dtype=np.float32)
                if names
                else np.array([0.3, 0.7], dtype=np.float32)
            ),
        )
        self.m_space = SimpleNamespace(optimize_params=names, default=lambda: default)
        self.fitness_fn = fitness_fn
        self.calls = 0

    def step(self, population):
        self.calls += 1
        return None, self.fitness_fn(population), False, False, {}


@pytest.fixture
def fake_reporter():
    calls = []
    reporter = SimpleNamespace(
        report=lambda metrics: calls.append(metrics), close=lambda: None
    )
    return reporter, calls


@pytest.mark.unit
class TestRun:
    def test_requires_env(self):
        with pytest.raises(RuntimeError, match="RegulatorEnv"):
            make_es().run()

    def test_template_dimension_mismatch(self):
        opt = make_es(dimension=2)
        with pytest.raises(ValueError, match="does not match"):
            opt.env = _FakeRegulatorEnv(3, lambda pop: np.zeros(len(pop)))

    def test_generation_lifecycle(self, fake_reporter):
        reporter, calls = fake_reporter
        opt = make_es(dimension=2, pop_size=4, sigma_lr=0.0)
        opt.reporting = reporter
        opt.env = _FakeRegulatorEnv(2, lambda pop: -np.sum((pop - 0.8) ** 2, axis=1))

        result = opt.run()
        assert opt.generation == 1
        assert len(opt.population_history) == 1
        assert np.isfinite(result["best_fitness"])
        assert set(result) >= {"best_fitness", "population_history"}
        assert len(calls) == 1  # one report per generation
        peeked = calls[0]
        assert (
            peeked.generation == [1] and len(peeked.by_mechanism) == 4
        )  # logged after increment
        assert set(peeked.search_mean) == set(opt.parameter_names)

    def test_empty_fitness_skips_update(self, fake_reporter):
        reporter, _ = fake_reporter
        opt = make_es(dimension=2, pop_size=4)
        opt.reporting = reporter
        opt.env = _FakeRegulatorEnv(2, lambda pop: np.empty(0))
        result = opt.run()
        assert result["converged"] is False and opt.generation == 0

    def test_non_finite_and_mismatched_fitness_raise(self, fake_reporter):
        reporter, _ = fake_reporter
        opt = make_es(dimension=2, pop_size=4)
        opt.reporting = reporter
        opt.env = _FakeRegulatorEnv(2, lambda pop: np.array([1.0, np.nan, 1.0, 1.0]))
        with pytest.raises(RuntimeError, match="Non-finite"):
            opt.run()
        opt.env = _FakeRegulatorEnv(2, lambda pop: np.ones(3))
        with pytest.raises(RuntimeError, match="fitness values"):
            opt.run()

    def test_fixed_mode_run_plots_template_vector(self, fake_reporter):
        reporter, calls = fake_reporter
        opt = make_es(dimension=0, pop_size=2)
        opt.reporting = reporter
        opt.env = _FakeRegulatorEnv(0, lambda pop: np.array([1.0, 2.0]))
        opt.run()
        peeked = calls[0]
        assert set(peeked.search_mean) == {"a", "b"}  # default mechanism vector names
        assert peeked.search_mean["a"].value == pytest.approx([0.3])
        assert peeked.by_mechanism["1"].by_parameter["b"].value == pytest.approx([0.7])
        assert opt.best_fitness == 2.0


@pytest.mark.unit
class TestConfigTraining:
    def test_every_training_keyword_is_stored(self):
        cfg = ESConfig().training(
            sigma=0.2,
            mean_lr=0.3,
            sigma_lr=0.4,
            sigma_decay=0.9,
            min_sigma=0.01,
            max_sigma=0.4,
            generation=7,
            break_symmetry=True,
            convergence_eps=1e-3,
            convergence_patience=3,
            initial_mean=[0.1, 0.9],
        )
        assert (cfg.sigma, cfg.mean_lr, cfg.sigma_lr, cfg.sigma_decay) == (
            0.2,
            0.3,
            0.4,
            0.9,
        )
        assert (cfg.min_sigma, cfg.max_sigma) == (0.01, 0.4)
        assert cfg.break_symmetry is True
        assert (cfg.convergence_eps, cfg.convergence_patience) == (1e-3, 3)
        assert cfg.initial_mean == [0.1, 0.9]
        # ``generation`` is accepted and stored, but no attribute is declared for it
        # in ``__init__`` and the optimizer never reads it (see report).
        assert cfg.generation == 7

    def test_none_keeps_defaults_and_unknown_keywords_are_dropped(self):
        cfg = ESConfig().training(typo_sigma=1.0)
        assert cfg.sigma == 0.15 and not hasattr(cfg, "typo_sigma")
        assert not hasattr(cfg, "generation")
        assert cfg.training() is cfg  # fluent


@pytest.mark.unit
class TestSingleCandidateSampling:
    def test_first_sample_is_the_mean_then_one_perturbed_offspring(self):
        opt = make_es(dimension=2, pop_size=1, sigma=0.3)
        first = opt._sample_population()
        np.testing.assert_allclose(first, opt.mean[None, :])  # parent evaluated first
        opt._update_parameters(first, [1.0])
        offspring = opt._sample_population()  # one independent (non-mirrored) sample
        assert offspring.shape == (1, 2) and offspring.dtype == np.float32
        assert np.all((offspring > 0.0) & (offspring < 1.0))
        assert not np.allclose(offspring[0], opt.mean)

    def test_zero_sigma_lr_keeps_sigma_fixed(self):
        opt = make_es(dimension=2, pop_size=1, sigma=0.2, sigma_lr=0.0)
        opt._update_parameters(opt._sample_population(), [1.0])
        sigma0 = opt.sigma
        opt._update_parameters(np.array([[0.2, 0.2]], dtype=np.float32), [0.0])
        assert opt.sigma == sigma0  # rejected, no contraction
        opt._update_parameters(np.array([[0.8, 0.8]], dtype=np.float32), [2.0])
        assert opt.sigma == sigma0  # accepted, no expansion
        assert opt.previous_population_mean_fitness == 2.0

    def test_direct_single_update_rejects_non_finite(self):
        # ``_update_parameters`` filters non-finite values before dispatching, so
        # this guard is only reachable by calling the (1+1) update directly.
        opt = make_es(dimension=2, pop_size=1)
        with pytest.raises(ValueError, match="must be finite"):
            opt._update_single_candidate(opt.mean, float("nan"))


@pytest.mark.unit
class TestFixedModeBatches:
    def test_second_batch_without_improvement_keeps_best(self):
        opt = make_es(dimension=0, pop_size=2)
        opt._update_parameters(np.empty((2, 0)), [1.0, 5.0])
        assert opt.best_fitness == 5.0 and opt.best_mechanism_idx == 1
        opt._update_parameters(np.empty((2, 0)), [3.0, 2.0])
        assert opt.best_fitness == 5.0 and opt.best_mechanism_idx == 1
        assert opt.fitness_baseline == pytest.approx(2.5)
        assert opt.previous_population_mean_fitness == pytest.approx(2.5)

    def test_fixed_mode_sampling_is_empty_and_run_twice(self, fake_reporter):
        reporter, calls = fake_reporter
        opt = make_es(dimension=0, pop_size=3)
        assert opt._sample_population().shape == (3, 0)
        opt.reporting = reporter
        opt.env = _FakeRegulatorEnv(0, lambda pop: np.array([1.0, 2.0, 0.5]))
        opt.run()
        opt.run()
        assert opt.generation == 2 and len(calls) == 2
        assert calls[1].generation == [1, 2]
        assert calls[1].best_mechanism_idx == [1, 1]
