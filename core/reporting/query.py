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

    def __post_init__(self) -> None:
        if self.error != "none" and self.reduce == "none":
            raise ValueError(
                "Query error requires a reduction. For example: reduce='mean', error='std'."
            )
        if not self.x:
            raise ValueError("Query x path must not be empty.")
        if not self.y_paths or any(not p for p in self.y_paths):
            raise ValueError("Query y paths must not be empty.")

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
    """

    label: str
    x: list
    y: list
    error: list | None = None
