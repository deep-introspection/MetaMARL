from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import wandb

from core.utils import sanitize_key


# One accumulated history table for each W&B run and reporting prefix.
_ES_HISTORY_TABLES: dict[tuple[int, str], wandb.Table] = {}


def _validate_inputs(
    *,
    population: np.ndarray,
    fitness: np.ndarray,
    parameter_names: Sequence[str],
) -> tuple[np.ndarray, float]:
    """
    Validate one evaluated mechanism and its scalar fitness.

    Expected shapes
    ---------------
    population:
        [1, mechanism_dimension]

    fitness:
        [1] or a scalar

    The single row in ``population`` is one complete mechanism, and
    ``fitness`` is the single scalar objective value assigned to it.
    """
    population = np.asarray(population, dtype=np.float64)
    fitness_array = np.asarray(fitness, dtype=np.float64).reshape(-1)

    if population.ndim != 2:
        raise ValueError(
            "ES population must have shape [1, mechanism_dimension], "
            f"got shape={population.shape}"
        )

    if population.shape[0] != 1:
        raise ValueError(
            "This reporter expects exactly one evaluated mechanism per "
            "outer optimization iteration, "
            f"got population_size={population.shape[0]}"
        )

    if population.shape[1] == 0:
        raise ValueError("ES mechanism dimension cannot be zero")

    if fitness_array.size != 1:
        raise ValueError(
            "This reporter expects exactly one scalar fitness value per "
            "evaluated mechanism, "
            f"got fitness shape={fitness_array.shape}"
        )

    if population.shape[1] != len(parameter_names):
        raise ValueError(
            "Parameter-name count does not match ES dimension: "
            f"dimension={population.shape[1]}, "
            f"parameter_names={len(parameter_names)}"
        )

    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("ES parameter names must be unique")

    if not np.all(np.isfinite(population)):
        raise ValueError("ES population contains NaN or infinite values")

    if not np.all(np.isfinite(fitness_array)):
        raise ValueError("ES fitness contains NaN or infinite values")

    return population, float(fitness_array[0])


def _validate_optional_vector(
    *,
    value: Optional[np.ndarray],
    expected_dimension: int,
    name: str,
) -> Optional[np.ndarray]:
    """Normalize and validate an optional mechanism-parameter vector."""
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float64).reshape(-1)

    if array.shape[0] != expected_dimension:
        raise ValueError(
            f"{name} dimension does not match population dimension: "
            f"{name}={array.shape[0]}, "
            f"population_dimension={expected_dimension}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")

    return array


def _get_or_create_history_table(
    *,
    wandb_run: Any,
    prefix: str,
    parameter_names: Sequence[str],
) -> wandb.Table:
    """
    Return the accumulated mechanism history table for this W&B run.

    Every row represents one completed outer optimization iteration:
    one complete mechanism and its single scalar fitness.
    """
    cache_key = (id(wandb_run), prefix)
    table = _ES_HISTORY_TABLES.get(cache_key)

    expected_columns = [
        "generation",
        "fitness",
        "sigma",
        *parameter_names,
    ]

    if table is None:
        table = wandb.Table(columns=expected_columns)
        _ES_HISTORY_TABLES[cache_key] = table
        return table

    if list(table.columns) != expected_columns:
        raise ValueError(
            "ES parameter names changed during the W&B run. "
            f"Existing columns={list(table.columns)}, "
            f"requested columns={expected_columns}"
        )

    return table


def _make_fitness_over_generations_figure(
    history_table: wandb.Table,
) -> go.Figure:
    """
    Plot the scalar mechanism fitness over outer optimization iterations.

    Each history-table row contributes exactly one point.
    """
    columns = list(history_table.columns)
    generation_col = columns.index("generation")
    fitness_col = columns.index("fitness")

    ordered_rows = sorted(
        history_table.data,
        key=lambda row: int(row[generation_col]),
    )

    generations = [int(row[generation_col]) for row in ordered_rows]
    fitness_values = [float(row[fitness_col]) for row in ordered_rows]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=generations,
            y=fitness_values,
            mode="lines+markers",
            name="Mechanism fitness",
            hovertemplate=(
                "outer iteration=%{x}"
                "<br>fitness=%{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Fitness over outer optimization iterations",
        xaxis_title="outer optimization iteration",
        yaxis_title="objective fitness",
        template="plotly_white",
        hovermode="x unified",
        height=600,
    )
    figure.update_xaxes(rangeslider_visible=False)

    return figure

def _padded_range(
    values: Sequence[float],
    *,
    minimum_padding: float = 1e-3,
    padding_fraction: float = 0.12,
) -> list[float]:
    values_array = np.asarray(values, dtype=np.float64)

    value_min = float(np.min(values_array))
    value_max = float(np.max(values_array))
    value_span = value_max - value_min

    padding = max(
        minimum_padding,
        padding_fraction * value_span,
    )

    if value_span == 0.0:
        padding = max(
            minimum_padding,
            abs(value_min) * padding_fraction,
        )

    return [
        value_min - padding,
        value_max + padding,
    ]


def _make_parameter_fitness_figure(
    *,
    history_table: wandb.Table,
    parameter_name: str,
) -> go.Figure:
    """
    Plot accumulated fitness versus one mechanism parameter.

    Every point represents one complete mechanism evaluated by the outer
    optimizer. The plot accumulates points across all completed iterations.
    """
    columns = list(history_table.columns)
    generation_col = columns.index("generation")
    fitness_col = columns.index("fitness")
    parameter_col = columns.index(parameter_name)

    parameter_values: list[float] = []
    fitness_values: list[float] = []
    hover_data: list[list[int]] = []

    for row in history_table.data:
        generation = int(row[generation_col])
        mechanism_fitness = float(row[fitness_col])
        parameter_value = float(row[parameter_col])

        parameter_values.append(parameter_value)
        fitness_values.append(mechanism_fitness)
        hover_data.append([generation])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=parameter_values,
            y=fitness_values,
            mode="markers",
            name="Evaluated mechanisms",
            customdata=hover_data,
            marker={
                "size": 11,
                "opacity": 0.8,
            },
            hovertemplate=(
                "outer iteration=%{customdata[0]}"
                "<br>parameter=%{x:.6f}"
                "<br>fitness=%{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": f"Fitness vs {parameter_name}",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=parameter_name,
        yaxis_title="objective fitness",
        template="plotly_white",
        hovermode="closest",
        height=650,
        margin={
            "l": 70,
            "r": 25,
            "t": 70,
            "b": 70,
        },
    )

    figure.update_xaxes(
        rangeslider_visible=False,
        automargin=True,
    )

    figure.update_yaxes(
        range=_padded_range(fitness_values),
        automargin=True,
        zeroline=True,
        zerolinewidth=1,
    )

    return figure


def plot_es_population(
    *,
    wandb_run: Any,
    generation: int,
    population: np.ndarray,
    fitness: np.ndarray,
    parameter_names: Sequence[str],
    mean: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
    best_fitness_global: Optional[float] = None,
    best_candidate_global: Optional[np.ndarray] = None,
    prefix: str = "es",
) -> None:
    """
    Log one completed outer optimization iteration to W&B.

    Parameters
    ----------
    wandb_run:
        Active Weights & Biases run.

    generation:
        Current outer optimization iteration. The existing argument name is
        retained for compatibility with the optimizer call site.

    population:
        One complete mechanism with shape ``[1, mechanism_dimension]``.

    fitness:
        The mechanism's single scalar objective value, supplied as a scalar
        or a one-element array such as ``array([-0.09730779])``.

    parameter_names:
        Mechanism parameter names in the same order as the mechanism vector.

    mean:
        Optional ES search-distribution mean. Retained for compatibility and
        logged once per parameter when provided.

    sigma:
        Optional ES exploration scale.

    best_fitness_global:
        Optional best fitness observed across all completed outer iterations.

    best_candidate_global:
        Optional mechanism vector associated with the global-best fitness.

    Logged plots
    ------------
    - Fitness over outer optimization iterations.
    - One cumulative fitness-versus-parameter scatter plot per mechanism
      parameter.

    Every completed outer iteration contributes one mechanism, one fitness
    value, and one point to each parameter scatter plot.
    """
    if wandb_run is None:
        return

    population, mechanism_fitness = _validate_inputs(
        population=population,
        fitness=fitness,
        parameter_names=parameter_names,
    )

    generation = int(generation)
    mechanism_dimension = population.shape[1]
    mechanism = population[0]

    mean_array = _validate_optional_vector(
        value=mean,
        expected_dimension=mechanism_dimension,
        name="ES search mean",
    )

    global_best_array = _validate_optional_vector(
        value=best_candidate_global,
        expected_dimension=mechanism_dimension,
        name="ES global-best candidate",
    )

    if sigma is not None:
        sigma = float(sigma)
        if not np.isfinite(sigma):
            raise ValueError("ES sigma contains NaN or an infinite value")

    if best_fitness_global is not None:
        best_fitness_global = float(best_fitness_global)
        if not np.isfinite(best_fitness_global):
            raise ValueError(
                "Global-best fitness contains NaN or an infinite value"
            )

    history_table = _get_or_create_history_table(
        wandb_run=wandb_run,
        prefix=prefix,
        parameter_names=parameter_names,
    )

    history_table.add_data(
        generation,
        mechanism_fitness,
        sigma,
        *mechanism.astype(float).tolist(),
    )

    payload: dict[str, Any] = {
        f"{prefix}/generation": generation,
        f"{prefix}/fitness": mechanism_fitness,
        f"{prefix}/tables/all_generations": history_table,
        f"{prefix}/plots/fitness_over_generations": (
            _make_fitness_over_generations_figure(history_table)
        ),
    }

    if sigma is not None:
        payload[f"{prefix}/sigma"] = sigma

    if best_fitness_global is not None:
        payload[f"{prefix}/best_fitness_global"] = best_fitness_global

    for parameter_idx, parameter_name in enumerate(parameter_names):
        clean_name = sanitize_key(parameter_name)

        payload[
            f"{prefix}/plots/fitness_vs_parameter/{clean_name}"
        ] = _make_parameter_fitness_figure(
            history_table=history_table,
            parameter_name=parameter_name,
        )

        if mean_array is not None:
            payload[f"{prefix}/search_mean/{clean_name}"] = float(
                mean_array[parameter_idx]
            )

        if global_best_array is not None:
            payload[f"{prefix}/global_best/{clean_name}"] = float(
                global_best_array[parameter_idx]
            )

    # One atomic W&B log operation for the completed outer iteration.
    wandb_run.log(
        payload,
        commit=True,
    )
