"""``ESOptimizer._to_logger_payload`` and the ES metric schema (TODO §5.1, §13)."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.es.config import ESConfig
from core.optimizers.es.optimizer import ESOptimizer
from core.optimizers.es.schema import ESSchema


class InnerSchema(MetricSchema):
    value: float | None = None


def make_es(dimension=2, pop_size=4):
    cfg = ESConfig().training(sigma=0.1)
    cfg.dimension = dimension
    cfg.debugging(seed=0)
    opt = ESOptimizer(config=cfg)
    opt.batch_capacity = pop_size
    opt.parameter_names = ["fixed_quota", "restoration_subsidy"][:dimension]
    return opt


@pytest.mark.unit
def test_schema_has_generation_fields():
    fields = ESSchema.model_fields
    assert "generation" in fields and "generation_best" in fields
    assert fields["generation"].json_schema_extra["reduce"].value == "series"


@pytest.mark.unit
def test_payload_population_mode_and_series_growth():
    opt = make_es()
    logger = MetricLogger.from_schema(ESSchema)
    population = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
    )
    fitness_by_gen = [np.array([1.0, 4.0, 2.0, 3.0]), np.array([2.0, 1.0, 5.0, 0.0])]

    for gen, fitness in enumerate(fitness_by_gen):
        opt.generation = gen
        opt.best_fitness = float(max(opt.best_fitness, fitness.max()))
        opt.best_candidate = population[int(np.argmax(fitness))]
        payload = opt._to_logger_payload(
            inner=InnerSchema(value=float(gen)),
            population=population,
            fitness=fitness,
            mean=np.array([0.4, 0.5]),
            sigma=0.1,
        )
        assert payload.generation == gen and payload.population_size == 4
        assert (
            payload.fitness_best == fitness.max()
            and payload.best_mechanism_idx == int(np.argmax(fitness))
        )
        assert set(payload.by_mechanism) == {"0", "1", "2", "3"}
        assert payload.by_mechanism["1"].by_parameter[
            "fixed_quota"
        ].value == pytest.approx(0.3)
        assert set(payload.search_mean) == {"fixed_quota", "restoration_subsidy"}
        best = int(np.argmax(fitness))
        assert payload.generation_best["restoration_subsidy"].value == pytest.approx(
            population[best, 1]
        )
        logger.push_data(payload)

    peeked = logger.peek()
    assert peeked.generation == [0, 1]
    assert peeked.fitness_best == [4.0, 5.0]
    assert peeked.best_fitness_global == [4.0, 5.0]
    assert peeked.by_mechanism["2"].fitness == [2.0, 5.0]
    assert peeked.by_mechanism["2"].by_parameter["fixed_quota"].value == [0.5, 0.5]
    assert peeked.generation_best["fixed_quota"].value == pytest.approx([0.3, 0.5])
    assert isinstance(peeked.inner, InnerSchema) and peeked.inner.value == [
        0.0,
        1.0,
    ]  # not reset


@pytest.mark.unit
def test_payload_fixed_mode_uses_default_mechanism_vector():
    opt = make_es(dimension=0, pop_size=2)
    mechanism = SimpleNamespace(
        param_names=lambda: ["a", "b"],
        to_vector=lambda: np.array([0.25, 0.75], dtype=np.float32),
    )
    opt.env = SimpleNamespace(
        m_space=SimpleNamespace(default=lambda: mechanism, optimize_params=[])
    )
    payload = opt._to_logger_payload(
        inner=InnerSchema(),
        population=np.empty((2, 0)),
        fitness=np.array([1.0, 2.0]),
        mean=np.empty(0),
        sigma=0.1,
    )
    assert payload.by_mechanism["0"].by_parameter["b"].value == pytest.approx(0.75)
    assert payload.search_mean["a"].value == pytest.approx(0.25)
    assert payload.global_best["b"].value == pytest.approx(0.75)
