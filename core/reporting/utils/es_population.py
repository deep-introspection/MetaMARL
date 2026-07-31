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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate one evaluated ES population and its fitness values.

    Expected shapes
    ---------------
    population:
        [population_size, mechanism_dimension]

    fitness:
        [population_size]

    Row ``i`` of the population must correspond to fitness value ``i``.
    """
    population = np.asarray(population, dtype=np.float64)
    fitness_array = np.asarray(fitness, dtype=np.float64).reshape(-1)

    if population.ndim != 2:
        raise ValueError(
            "ES population must have shape "
            "[population_size, mechanism_dimension], "
            f"got shape={population.shape}"
        )

    population_size, mechanism_dimension = population.shape

    if population_size == 0:
        raise ValueError("ES population cannot be empty")

    if mechanism_dimension == 0:
        raise ValueError("ES mechanism dimension cannot be zero")

    if fitness_array.size != population_size:
        raise ValueError(
            "Each evaluated mechanism must have one corresponding fitness: "
            f"population_size={population_size}, "
            f"fitness_count={fitness_array.size}"
        )

    if mechanism_dimension != len(parameter_names):
        raise ValueError(
            "Parameter-name count does not match ES dimension: "
            f"dimension={mechanism_dimension}, "
            f"parameter_names={len(parameter_names)}"
        )

    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("ES parameter names must be unique")

    if not np.all(np.isfinite(population)):
        raise ValueError("ES population contains NaN or infinite values")

    if not np.all(np.isfinite(fitness_array)):
        raise ValueError("ES fitness contains NaN or infinite values")

    return population, fitness_array


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

    Every row represents one evaluated mechanism from one outer generation.
    """
    cache_key = (id(wandb_run), prefix)
    table = _ES_HISTORY_TABLES.get(cache_key)

    expected_columns = [
        "generation",
        "mechanism_idx",
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
            "ES history-table columns changed during the W&B run. "
            f"Existing columns={list(table.columns)}, "
            f"requested columns={expected_columns}"
        )

    return table


def _make_fitness_over_generations_figure(
    history_table: wandb.Table,
) -> go.Figure:
    """
    Plot candidate, mean, and best fitness over outer generations.
    """
    columns = list(history_table.columns)
    generation_col = columns.index("generation")
    mechanism_idx_col = columns.index("mechanism_idx")
    fitness_col = columns.index("fitness")

    ordered_rows = sorted(
        history_table.data,
        key=lambda row: (
            int(row[generation_col]),
            int(row[mechanism_idx_col]),
        ),
    )

    fitness_by_generation: dict[int, list[float]] = {}

    candidate_generations: list[int] = []
    candidate_indices: list[int] = []
    candidate_fitness: list[float] = []

    for row in ordered_rows:
        generation = int(row[generation_col])
        mechanism_idx = int(row[mechanism_idx_col])
        mechanism_fitness = float(row[fitness_col])

        fitness_by_generation.setdefault(generation, []).append(
            mechanism_fitness
        )

        candidate_generations.append(generation)
        candidate_indices.append(mechanism_idx)
        candidate_fitness.append(mechanism_fitness)

    generations = sorted(fitness_by_generation)

    mean_fitness = [
        float(np.mean(fitness_by_generation[generation]))
        for generation in generations
    ]

    best_fitness = [
        float(np.max(fitness_by_generation[generation]))
        for generation in generations
    ]

    candidate_customdata = np.asarray(
        candidate_indices,
        dtype=np.int64,
    ).reshape(-1, 1)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=candidate_generations,
            y=candidate_fitness,
            mode="markers",
            name="Candidates",
            customdata=candidate_customdata,
            marker={
                "size": 8,
                "opacity": 0.35,
            },
            hovertemplate=(
                "outer iteration=%{x}"
                "<br>mechanism idx=%{customdata[0]}"
                "<br>fitness=%{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=generations,
            y=mean_fitness,
            mode="lines+markers",
            name="Generation mean",
            hovertemplate=(
                "outer iteration=%{x}"
                "<br>mean fitness=%{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=generations,
            y=best_fitness,
            mode="lines+markers",
            name="Generation best",
            hovertemplate=(
                "outer iteration=%{x}"
                "<br>best fitness=%{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title="Fitness over outer optimization iterations",
        xaxis_title="outer optimization iteration",
        yaxis_title="objective fitness",
        template="plotly_white",
        hovermode="closest",
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

    Every point represents one evaluated mechanism. Point colour represents
    its outer generation.
    """
    columns = list(history_table.columns)
    generation_col = columns.index("generation")
    mechanism_idx_col = columns.index("mechanism_idx")
    fitness_col = columns.index("fitness")
    parameter_col = columns.index(parameter_name)

    parameter_values: list[float] = []
    fitness_values: list[float] = []
    generations: list[int] = []
    mechanism_indices: list[int] = []

    ordered_rows = sorted(
        history_table.data,
        key=lambda row: (
            int(row[generation_col]),
            int(row[mechanism_idx_col]),
        ),
    )

    for row in ordered_rows:
        generations.append(int(row[generation_col]))
        mechanism_indices.append(int(row[mechanism_idx_col]))
        fitness_values.append(float(row[fitness_col]))
        parameter_values.append(float(row[parameter_col]))

    generation_min = min(generations)
    generation_max = max(generations)

    customdata = np.column_stack(
        [
            np.asarray(generations, dtype=np.int64),
            np.asarray(mechanism_indices, dtype=np.int64),
        ]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=parameter_values,
            y=fitness_values,
            mode="markers",
            name="Evaluated mechanisms",
            customdata=customdata,
            marker={
                "size": 11,
                "opacity": 0.85,
                "color": generations,
                "colorscale": "Viridis",
                "cmin": generation_min,
                "cmax": generation_max,
                "showscale": True,
                "colorbar": {
                    "title": {
                        "text": "Outer<br>iteration",
                    },
                    "thickness": 18,
                    "len": 0.85,
                },
            },
            hovertemplate=(
                "outer iteration=%{customdata[0]}"
                "<br>mechanism idx=%{customdata[1]}"
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
            "r": 95,
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

def _make_parallel_coordinates_figure(
    *,
    history_table: wandb.Table,
    parameter_names: Sequence[str],
) -> go.Figure:
    """
    Create a cumulative parallel-coordinates plot.

    Each line represents one evaluated mechanism.
    Each vertical axis represents one mechanism parameter.
    Line colour represents the mechanism's fitness.
    """
    columns = list(history_table.columns)

    generation_col = columns.index("generation")
    mechanism_idx_col = columns.index("mechanism_idx")
    fitness_col = columns.index("fitness")

    parameter_cols = {
        parameter_name: columns.index(parameter_name)
        for parameter_name in parameter_names
    }

    # Keep the mechanisms ordered by outer iteration.
    ordered_rows = sorted(
        history_table.data,
        key=lambda row: (
            int(row[generation_col]),
            int(row[mechanism_idx_col]),
        ),
    )

    fitness_values = np.asarray(
        [float(row[fitness_col]) for row in ordered_rows],
        dtype=np.float64,
    )

    if fitness_values.size == 0:
        return go.Figure()

    fitness_min = float(np.min(fitness_values))
    fitness_max = float(np.max(fitness_values))

    # Plotly requires a non-zero colour range.
    if fitness_min == fitness_max:
        colour_padding = max(
            1e-6,
            abs(fitness_min) * 0.01,
        )
        colour_min = fitness_min - colour_padding
        colour_max = fitness_max + colour_padding
    else:
        colour_min = fitness_min
        colour_max = fitness_max

    dimensions: list[dict[str, Any]] = []

    for parameter_name in parameter_names:
        parameter_col = parameter_cols[parameter_name]

        parameter_values = np.asarray(
            [
                float(row[parameter_col])
                for row in ordered_rows
            ],
            dtype=np.float64,
        )

        parameter_min = float(np.min(parameter_values))
        parameter_max = float(np.max(parameter_values))

        # Ensure the axis remains visible when all values are identical.
        if parameter_min == parameter_max:
            axis_padding = max(
                1e-6,
                abs(parameter_min) * 0.01,
            )
            axis_range = [
                parameter_min - axis_padding,
                parameter_max + axis_padding,
            ]
        else:
            axis_range = [
                parameter_min,
                parameter_max,
            ]

        dimensions.append(
            {
                "label": parameter_name,
                "values": parameter_values,
                "range": axis_range,
            }
        )

    # Add fitness as the final vertical axis.
    dimensions.append(
        {
            "label": "fitness",
            "values": fitness_values,
            "range": [colour_min, colour_max],
        }
    )

    figure = go.Figure(
        data=[
            go.Parcoords(
                line={
                    "color": fitness_values,
                    "colorscale": "Viridis",
                    "cmin": colour_min,
                    "cmax": colour_max,
                    "showscale": True,
                    "colorbar": {
                        "title": {
                            "text": "Fitness<br>(higher is better)",
                        },
                        "thickness": 18,
                        "len": 0.85,
                    },
                },
                dimensions=dimensions,
                labelfont={
                    "size": 13,
                },
                tickfont={
                    "size": 11,
                },
                rangefont={
                    "size": 10,
                },
            )
        ]
    )

    figure.update_layout(
        title={
            "text": "Parallel coordinates of evaluated mechanisms",
            "x": 0.5,
            "xanchor": "center",
        },
        template="plotly_white",
        height=750,
        margin={
            "l": 90,
            "r": 120,
            "t": 100,
            "b": 70,
        },
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

    population, fitness_array = _validate_inputs(
        population=population,
        fitness=fitness,
        parameter_names=parameter_names,
    )

    generation = int(generation)
    population_size, mechanism_dimension = population.shape

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
            raise ValueError(
                "ES sigma contains NaN or an infinite value"
            )

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

    # The regulator returns fitness indexed by mechanism index, so:
    #
    # population[mechanism_idx] corresponds to fitness_array[mechanism_idx].
    for mechanism_idx in range(population_size):
        history_table.add_data(
            generation,
            mechanism_idx,
            float(fitness_array[mechanism_idx]),
            sigma,
            *population[mechanism_idx].astype(float).tolist(),
        )

    generation_mean_fitness = float(np.mean(fitness_array))
    generation_best_position = int(np.argmax(fitness_array))
    generation_best_fitness = float(
        fitness_array[generation_best_position]
    )

    payload: dict[str, Any] = {
        f"{prefix}/generation": generation,
        f"{prefix}/population_size": population_size,
        f"{prefix}/fitness_mean": generation_mean_fitness,
        f"{prefix}/fitness_best": generation_best_fitness,
        f"{prefix}/best_mechanism_idx": generation_best_position,
        f"{prefix}/tables/all_generations": history_table,
        f"{prefix}/plots/fitness_over_generations": wandb.Plotly(
            _make_fitness_over_generations_figure(history_table)
        ),
        f"{prefix}/plots/parallel_coordinates": wandb.Plotly(
            _make_parallel_coordinates_figure(
                history_table=history_table,
                parameter_names=parameter_names,
            )
        ),
    }

    if sigma is not None:
        payload[f"{prefix}/sigma"] = sigma

    if best_fitness_global is not None:
        payload[
            f"{prefix}/best_fitness_global"
        ] = best_fitness_global

    for parameter_idx, parameter_name in enumerate(parameter_names):
        clean_name = sanitize_key(parameter_name)

        payload[
            f"{prefix}/plots/fitness_vs_parameter/{clean_name}"
        ] = wandb.Plotly(
            _make_parameter_fitness_figure(
                history_table=history_table,
                parameter_name=parameter_name,
            )
        )

        if mean_array is not None:
            payload[f"{prefix}/search_mean/{clean_name}"] = float(
                mean_array[parameter_idx]
            )

        if global_best_array is not None:
            payload[f"{prefix}/global_best/{clean_name}"] = float(
                global_best_array[parameter_idx]
            )

        # Log the best candidate from this generation separately.
        payload[
            f"{prefix}/generation_best/{clean_name}"
        ] = float(
            population[
                generation_best_position,
                parameter_idx,
            ]
        )

    # One atomic W&B log operation for the completed outer iteration.
    wandb_run.log(
        payload,
        commit=True,
    )
