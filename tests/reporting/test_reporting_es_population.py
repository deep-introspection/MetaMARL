"""Unit tests for ``core.reporting.utils.es_population``.

``plot_es_population`` validates one evaluated ES population, appends it to a
per-run history table and logs a fitness-over-generations figure, a
parallel-coordinates figure and one fitness-versus-parameter figure per
mechanism parameter. The tests cover every validation branch, the padding
helper used for axis ranges, the figure builders and the accumulated logging
across generations.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

import wandb
from core.reporting.utils import es_population as mod

pytestmark = pytest.mark.unit

NAMES = ["fixed_quota", "custom_param"]


def _history_table(rows):
    table = wandb.Table(
        columns=["generation", "mechanism_idx", "fitness", "sigma", *NAMES]
    )
    for row in rows:
        table.add_data(*row)
    return table


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_validate_inputs_happy_path():
    pop, fit = mod._validate_inputs(
        population=[[1, 2], [3, 4]], fitness=[0.1, 0.2], parameter_names=NAMES
    )
    assert pop.shape == (2, 2) and pop.dtype == np.float64
    assert fit.tolist() == [0.1, 0.2]


@pytest.mark.parametrize(
    ("population", "fitness", "names", "message"),
    [
        ([1.0, 2.0], [0.1], NAMES, "must have shape"),
        (np.zeros((0, 2)), [], NAMES, "cannot be empty"),
        (np.zeros((2, 0)), [0.0, 0.0], [], "dimension cannot be zero"),
        ([[1.0, 2.0]], [0.1, 0.2], NAMES, "one corresponding fitness"),
        ([[1.0, 2.0]], [0.1], ["a"], "Parameter-name count"),
        ([[1.0, 2.0]], [0.1], ["a", "a"], "must be unique"),
        ([[np.nan, 2.0]], [0.1], NAMES, "population contains NaN"),
        ([[1.0, 2.0]], [np.inf], NAMES, "fitness contains NaN"),
    ],
)
def test_validate_inputs_errors(population, fitness, names, message):
    with pytest.raises(ValueError, match=message):
        mod._validate_inputs(
            population=population, fitness=fitness, parameter_names=names
        )


def test_validate_optional_vector():
    assert (
        mod._validate_optional_vector(value=None, expected_dimension=2, name="x")
        is None
    )
    out = mod._validate_optional_vector(value=[[1, 2]], expected_dimension=2, name="x")
    assert out.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="dimension does not match"):
        mod._validate_optional_vector(value=[1.0], expected_dimension=2, name="x")
    with pytest.raises(ValueError, match="contains NaN"):
        mod._validate_optional_vector(
            value=[1.0, np.nan], expected_dimension=2, name="x"
        )


def test_history_table_cache_and_column_check(fake_run):
    t1 = mod._get_or_create_history_table(
        wandb_run=fake_run, prefix="es", parameter_names=NAMES
    )
    t2 = mod._get_or_create_history_table(
        wandb_run=fake_run, prefix="es", parameter_names=NAMES
    )
    assert t1 is t2
    assert list(t1.columns) == [
        "generation",
        "mechanism_idx",
        "fitness",
        "sigma",
        *NAMES,
    ]
    with pytest.raises(ValueError, match="columns changed"):
        mod._get_or_create_history_table(
            wandb_run=fake_run, prefix="es", parameter_names=["other", "names"]
        )
    # a different prefix owns its own table
    t3 = mod._get_or_create_history_table(
        wandb_run=fake_run, prefix="es2", parameter_names=NAMES
    )
    assert t3 is not t1


# ---------------------------------------------------------------------------
# figure helpers
# ---------------------------------------------------------------------------


def test_padded_range_span_and_constant():
    lo, hi = mod._padded_range([0.0, 10.0])
    assert lo == pytest.approx(-1.2) and hi == pytest.approx(11.2)
    lo, hi = mod._padded_range([5.0, 5.0])
    assert lo == pytest.approx(5.0 - 0.6) and hi == pytest.approx(5.6)
    lo, hi = mod._padded_range([0.0, 0.0])
    assert lo == pytest.approx(-1e-3) and hi == pytest.approx(1e-3)


def test_fitness_over_generations_figure():
    table = _history_table(
        [
            [1, 1, 0.5, None, 1.0, 2.0],
            [0, 0, 0.1, None, 1.0, 2.0],
            [1, 0, 0.3, None, 1.5, 2.0],
        ]
    )
    fig = mod._make_fitness_over_generations_figure(table)
    assert isinstance(fig, go.Figure)
    candidates, mean, best = fig.data
    assert list(candidates.x) == [0, 1, 1]
    assert list(mean.x) == [0, 1]
    assert list(mean.y) == [0.1, pytest.approx(0.4)]
    assert list(best.y) == [0.1, 0.5]


def test_parameter_fitness_figure():
    table = _history_table([[0, 0, 0.1, 0.5, 1.0, 2.0], [1, 0, 0.3, 0.5, 1.5, 2.0]])
    fig = mod._make_parameter_fitness_figure(
        history_table=table, parameter_name="fixed_quota"
    )
    (trace,) = fig.data
    assert list(trace.x) == [1.0, 1.5]
    assert trace.marker.cmin == 0 and trace.marker.cmax == 1
    assert fig.layout.xaxis.title.text == "fixed_quota"


def test_parallel_coordinates_figure_empty_and_constant_fitness():
    empty = _history_table([])
    fig = mod._make_parallel_coordinates_figure(
        history_table=empty, parameter_names=NAMES
    )
    assert fig.data == ()

    constant = _history_table(
        [[0, 0, 0.2, None, 1.0, 2.0], [0, 1, 0.2, None, 1.5, 2.5]]
    )
    fig = mod._make_parallel_coordinates_figure(
        history_table=constant, parameter_names=NAMES
    )
    (parcoords,) = fig.data
    labels = [d["label"] for d in parcoords.dimensions]
    assert labels == ["Fixed quota", "Custom Param", "Fitness"]
    assert parcoords.line.cmin < 0.2 < parcoords.line.cmax


def test_parallel_coordinates_figure_varying_fitness():
    table = _history_table([[0, 0, 0.2, None, 1.0, 2.0], [0, 1, 0.8, None, 1.5, 2.5]])
    fig = mod._make_parallel_coordinates_figure(
        history_table=table, parameter_names=NAMES
    )
    (parcoords,) = fig.data
    assert parcoords.line.cmin == 0.2 and parcoords.line.cmax == 0.8


# ---------------------------------------------------------------------------
# plot_es_population
# ---------------------------------------------------------------------------


def test_plot_es_population_none_run_is_noop(fake_run):
    mod.plot_es_population(
        wandb_run=None,
        generation=0,
        population=[[1.0, 2.0]],
        fitness=[0.1],
        parameter_names=NAMES,
    )
    assert mod._ES_HISTORY_TABLES == {}


def test_plot_es_population_full_payload(fake_run):
    mod.plot_es_population(
        wandb_run=fake_run,
        generation=0,
        population=np.array([[1.0, 2.0], [1.5, 2.5]]),
        fitness=np.array([0.1, 0.4]),
        parameter_names=NAMES,
        mean=[1.2, 2.2],
        sigma=0.3,
        best_fitness_global=0.4,
        best_candidate_global=[1.5, 2.5],
        prefix="es",
    )
    assert len(fake_run.logs) == 1
    payload, step, commit = fake_run.logs[0]
    assert step is None and commit is True
    assert payload["es/generation"] == 0
    assert payload["es/population_size"] == 2
    assert payload["es/fitness_mean"] == pytest.approx(0.25)
    assert payload["es/fitness_best"] == 0.4
    assert payload["es/best_mechanism_idx"] == 1
    assert payload["es/sigma"] == 0.3
    assert payload["es/best_fitness_global"] == 0.4
    assert payload["es/search_mean/fixed_quota"] == 1.2
    assert payload["es/global_best/custom_param"] == 2.5
    assert payload["es/generation_best/fixed_quota"] == 1.5
    assert isinstance(payload["es/plots/fitness_over_generations"], wandb.Plotly)
    assert isinstance(payload["es/plots/parallel_coordinates"], wandb.Plotly)
    assert isinstance(
        payload["es/plots/fitness_vs_parameter/custom_param"], wandb.Plotly
    )
    assert (
        payload["es/tables/all_generations"]
        is mod._ES_HISTORY_TABLES[(id(fake_run), "es")]
    )


def test_plot_es_population_accumulates_and_omits_optionals(fake_run):
    for generation in range(3):
        mod.plot_es_population(
            wandb_run=fake_run,
            generation=generation,
            population=[[float(generation), 1.0]],
            fitness=[-0.1 * generation],
            parameter_names=NAMES,
        )
    table = mod._ES_HISTORY_TABLES[(id(fake_run), "es")]
    assert len(table.data) == 3
    payload = fake_run.logs[-1][0]
    assert "es/sigma" not in payload
    assert "es/best_fitness_global" not in payload
    assert not any(k.startswith("es/search_mean") for k in payload)
    assert not any(k.startswith("es/global_best") for k in payload)


def test_plot_es_population_rejects_non_finite_scalars(fake_run):
    kwargs = dict(
        wandb_run=fake_run,
        generation=0,
        population=[[1.0, 2.0]],
        fitness=[0.1],
        parameter_names=NAMES,
    )
    with pytest.raises(ValueError, match="sigma"):
        mod.plot_es_population(sigma=float("nan"), **kwargs)
    with pytest.raises(ValueError, match="Global-best fitness"):
        mod.plot_es_population(best_fitness_global=float("inf"), **kwargs)
    assert fake_run.logs == []
