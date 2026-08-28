"""Declarative selection of metric series to report.

A :class:`Query` names an x path and one or several y paths inside a
``MetricSchema`` tree. Paths are tuples of field names; at a *dynamic* node
(``dict[ID, MetricSchema]``) a component may be the wildcard ``"*"``, which
expands to every runtime key in sorted order::

    Query(title="Fish biomass by mechanism",
          x=("iter",),
          y=("train", "rollout", "by_mechanism", "*", "by_seed", "*", "fish_norm"),
          reduce="mean", error="std")

Wildcards bind the same way on the x and y sides, so a scatter such as
``x=("by_mechanism", "*", "by_parameter", "q", "value")`` against
``y=("by_mechanism", "*", "fitness")`` pairs candidate ``k`` with candidate
``k`` and never forms a Cartesian product.

With ``reduce="mean"`` and two or more wildcard levels, the expanded series
are grouped by the *first* binding and averaged over the remaining ones (one
mean per mechanism across seeds); with a single level, all matches are
averaged together (mean across candidates or agents); without wildcards, the
listed y paths are averaged. ``error="std"`` adds the standard deviation band.

``color`` names a third path resolved and aligned exactly like ``x`` (same
wildcard bindings, same length as ``y``); each point then carries one value
that a backend may render as a colour, e.g. the outer ``generation`` of an
ES candidate::

    Query(title="Fitness vs fixed_quota",
          x=("by_mechanism", "*", "by_parameter", "fixed_quota", "value"),
          y=("by_mechanism", "*", "fitness"),
          color=("generation",))

A :class:`ParallelCoordinatesQuery` selects several axes at once and resolves
to a :class:`Table` (one row per evaluated entity and index) rather than to
series; see :meth:`core.reporting.base.Reporter._resolve_parallel` for the
row and axis-label rules.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

Path: TypeAlias = tuple[str, ...]
WILDCARD = "*"


@dataclass(frozen=True, slots=True)
class Query:
    """Defines a reporting query over a MetricSchema (see module docstring)."""

    title: str
    x: Path
    y: Path | tuple[Path, ...]
    reduce: Literal["none", "mean"] = "none"
    error: Literal["none", "std"] = "none"
    color: Path | None = None

    def __post_init__(self) -> None:
        if self.error != "none" and self.reduce == "none":
            raise ValueError(
                "Query error requires a reduction. For example: reduce='mean', error='std'."
            )
        if not self.x:
            raise ValueError("Query x path must not be empty.")
        if not self.y_paths or any(not p for p in self.y_paths):
            raise ValueError("Query y paths must not be empty.")
        if self.color is not None:
            if not self.color:
                raise ValueError("Query color path must not be empty.")
            if self.reduce != "none":
                # A colour per point is meaningless once points are averaged.
                raise ValueError(
                    "Query color requires reduce='none': averaged points have no "
                    "per-point colour."
                )

    @property
    def y_paths(self) -> tuple[Path, ...]:
        if self.y and isinstance(self.y[0], tuple):
            return cast(tuple[Path, ...], self.y)
        return (cast(Path, self.y),)

    @property
    def has_wildcards(self) -> bool:
        return WILDCARD in self.x or any(WILDCARD in p for p in self.y_paths)


@dataclass(frozen=True, slots=True)
class Series:
    """One resolved series handed to a reporting backend.

    Parameters
    ----------
    label : str
        ``/``-joined metric path with wildcards replaced by the bound ids
        (or the group id for reduced series).
    x, y : list
        Aligned values.
    error : list, optional
        Standard deviation per point when the query asked for ``error="std"``.
    color : list, optional
        One value per point when the query named a ``color`` path.
    """

    label: str
    x: list
    y: list
    error: list | None = None
    color: list | None = None


@dataclass(frozen=True, slots=True)
class ParallelCoordinatesQuery:
    """Select several axes of a schema as one parallel-coordinates table.

    Parameters
    ----------
    title : str
        Figure title; also the logging key of the rendered table.
    dimensions : tuple of Path
        One path per axis, in axis order. A wildcard bound after the entity
        wildcard (``("by_mechanism", "*", "by_parameter", "*", "value")``)
        yields one axis per bound id.
    color : Path
        Path of the value colouring each line (resolved like a dimension).

    When to use
    -----------
    To compare every evaluated candidate across all optimized parameters and
    its fitness on one figure, accumulated over generations. Use :class:`Query`
    for anything that is a line or a scatter against one x axis.

    Examples
    --------
    >>> ParallelCoordinatesQuery(
    ...     title="Parallel coordinates of evaluated mechanisms",
    ...     dimensions=(("by_mechanism", "*", "by_parameter", "*", "value"),
    ...                 ("by_mechanism", "*", "fitness")),
    ...     color=("by_mechanism", "*", "fitness"),
    ... ).color
    ('by_mechanism', '*', 'fitness')
    """

    title: str
    dimensions: tuple[Path, ...]
    color: Path

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise ValueError("ParallelCoordinatesQuery needs at least one dimension.")
        if any(not p for p in self.dimensions):
            raise ValueError(
                "ParallelCoordinatesQuery dimension paths must not be empty."
            )
        if not self.color:
            raise ValueError("ParallelCoordinatesQuery color path must not be empty.")


AnyQuery: TypeAlias = Query | ParallelCoordinatesQuery
"""Every query type a :class:`core.reporting.base.Reporter` accepts."""


@dataclass(frozen=True, slots=True)
class Table:
    """One resolved :class:`ParallelCoordinatesQuery` handed to a backend.

    Parameters
    ----------
    columns : tuple of str
        One axis label per column, in dimension order.
    rows : list of list of float
        One row per evaluated entity and index (shape ``(n_rows, n_columns)``),
        index-major: all entities of index 0, then all entities of index 1, ...
    color : list of float
        One value per row (``len(color) == len(rows)``).
    color_label : str
        Axis label of the colour path (colourbar title).
    """

    columns: tuple[str, ...]
    rows: list[list[float]]
    color: list[float]
    color_label: str
