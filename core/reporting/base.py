"""Reporter base: resolve queries against a populated metric schema.

``Reporter.report(metrics)`` resolves every registered :class:`Query` into a
list of :class:`Series` — expanding wildcards, aligning x and y by their
wildcard bindings and applying the requested reduction — then hands them to
the backend-specific :meth:`Reporter._report`. Backends therefore never see
raw metric trees, only labeled ``(x, y[, error])`` series.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import numpy as np

from core.metrics.metric.base import PrimitiveType
from core.metrics.schemas import MetricSchema
from core.reporting.query import WILDCARD, Path, Query, Series

Bindings = tuple[str, ...]
"""Runtime ids bound to the wildcards of a path, in path order."""


class Reporter(ABC):
    """Base interface for reporting resolved metric series.

    Reporters are configured after construction (``schema`` is write-once,
    queries accumulate) and are driven by :meth:`report`.
    """

    _queries: tuple[Query, ...] = ()
    _schema: type[MetricSchema] | None = None

    # --- configuration -------------------------------------------------------------

    @property
    def queries(self) -> tuple[Query, ...]:
        """Queries registered with this reporter."""
        return self._queries

    def add_query(self, *queries: Query) -> None:
        """Register one or more reporting queries."""
        self._queries += queries

    @property
    def schema(self) -> type[MetricSchema] | None:
        return self._schema

    @schema.setter
    def schema(self, schema: type[MetricSchema]) -> None:
        if self._schema is not None:
            raise AttributeError(
                "Reporter schema has already been set and cannot be changed."
            )
        self._schema = schema

    # --- path resolution ---------------------------------------------------------------

    @staticmethod
    def _child(metrics: MetricSchema | dict, key: str, path: Path) -> Any:
        if isinstance(metrics, dict):
            try:
                return metrics[key]
            except KeyError:
                raise KeyError(f"Unknown metric path: {path} (no id {key!r})") from None
        try:
            return getattr(metrics, key)
        except AttributeError:
            raise KeyError(f"Unknown metric path: {path} (no field {key!r})") from None

    def _expand_path(
        self,
        path: Path,
        metrics: MetricSchema | dict,
        *,
        index: int = 0,
        bindings: Bindings = (),
    ) -> list[tuple[Bindings, list[PrimitiveType]]]:
        """Resolve ``path`` into ``[(bindings, series), ...]``, expanding wildcards.

        A wildcard is only valid at a dynamic node (a ``dict`` of schemas); the
        matched ids are appended to ``bindings`` in sorted order. The leaf must
        be a list (a series), otherwise ``TypeError`` is raised.
        """
        if index >= len(path):
            raise KeyError(f"Path does not point to a metric: {path}")
        key = path[index]

        if key == WILDCARD:
            if not isinstance(metrics, dict):
                raise KeyError(
                    f"Wildcard '*' is only valid at a dynamic node: {path} (index {index})"
                )
            expanded: list[tuple[Bindings, list[PrimitiveType]]] = []
            for runtime_id in sorted(metrics, key=str):
                expanded.extend(
                    self._expand_path(
                        path,
                        metrics[runtime_id],
                        index=index + 1,
                        bindings=bindings + (str(runtime_id),),
                    )
                )
            return expanded

        child = self._child(metrics, key, path)

        if index == len(path) - 1:
            if not isinstance(child, list):
                raise TypeError(f"Path does not point to a metric series: {path}")
            return [(bindings, child)]

        if not isinstance(child, (MetricSchema, dict)):
            raise KeyError(f"Path continues beyond metric leaf: {path}")
        return self._expand_path(path, child, index=index + 1, bindings=bindings)

    def _resolve_path(
        self,
        path: Path,
        metrics: MetricSchema | dict[str, MetricSchema],
        *,
        index: int = 0,
    ) -> list[PrimitiveType]:
        """Resolve a wildcard-free path to its series."""
        if WILDCARD in path:
            raise KeyError(f"_resolve_path does not expand wildcards: {path}")
        ((_, series),) = self._expand_path(path, metrics, index=index)
        return series

    # --- query resolution -----------------------------------------------------------

    @staticmethod
    def _label(path: Path, bindings: Bindings) -> str:
        parts: list[str] = []
        bound = iter(bindings)
        for component in path:
            parts.append(next(bound) if component == WILDCARD else component)
        return "/".join(parts)

    def _resolve_x(
        self, query: Query, metrics: MetricSchema
    ) -> dict[Bindings, list[PrimitiveType]]:
        return {
            bindings: values for bindings, values in self._expand_path(query.x, metrics)
        }

    @staticmethod
    def _x_for(
        x_by_binding: dict[Bindings, list[PrimitiveType]],
        y_bindings: Bindings,
        query: Query,
    ) -> list[PrimitiveType]:
        """Pick the x series sharing the y series' leading bindings (no Cartesian product)."""
        if len(x_by_binding) == 1 and () in x_by_binding:
            return x_by_binding[()]
        for x_bindings, values in x_by_binding.items():
            if y_bindings[: len(x_bindings)] == x_bindings:
                return values
        raise KeyError(
            f"No x series for bindings {y_bindings} in query {query.title!r}: "
            f"x={query.x} binds {sorted(x_by_binding)}"
        )

    def _resolve_query(self, metrics: MetricSchema, query: Query) -> list[Series]:
        """Resolve ``query`` against ``metrics`` into labeled series.

        Raises
        ------
        KeyError
            If a path does not exist.
        ValueError
            If x and y lengths differ, or replicates of a group differ in length.
        """
        x_by_binding = self._resolve_x(query, metrics)

        raw: list[tuple[Bindings, str, list[PrimitiveType], list[PrimitiveType]]] = []
        for path in query.y_paths:
            for bindings, values in self._expand_path(path, metrics):
                x = self._x_for(x_by_binding, bindings, query)
                if len(x) != len(values):
                    raise ValueError(
                        f"Query series must have equal length: x={query.x} ({len(x)}), "
                        f"y={self._label(path, bindings)} ({len(values)})."
                    )
                raw.append((bindings, self._label(path, bindings), x, values))

        if query.reduce == "none":
            return [Series(label=label, x=list(x), y=list(y)) for _, label, x, y in raw]

        if query.reduce != "mean":
            raise ValueError(f"Unknown query reduction: {query.reduce!r}")

        # Group by the first wildcard binding (e.g. one group per mechanism) and
        # average over the remaining ones (e.g. seeds). Without wildcards, the
        # listed y paths form a single group labeled by the query title.
        # A single wildcard level has nothing left to average once grouped, so
        # its matches are the replicates of one group (mean across candidates,
        # agents, ...). Two or more levels group by the first one.
        groups: "OrderedDict[str, list[tuple[list, list]]]" = OrderedDict()
        for bindings, _, x, y in raw:
            group = bindings[0] if len(bindings) >= 2 else query.title
            groups.setdefault(group, []).append((x, y))

        series: list[Series] = []
        for group, replicates in groups.items():
            x0 = replicates[0][0]
            if any(len(y) != len(x0) or list(x) != list(x0) for x, y in replicates):
                raise ValueError(
                    f"Replicates of group {group!r} in query {query.title!r} are not aligned."
                )
            values = np.asarray([y for _, y in replicates], dtype=np.float64)
            mean = values.mean(axis=0).tolist()
            error = values.std(axis=0).tolist() if query.error == "std" else None
            label = (
                group
                if groups and len(groups) > 1 or (raw and raw[0][0])
                else query.title
            )
            series.append(Series(label=label, x=list(x0), y=mean, error=error))
        return series

    # --- backend contract --------------------------------------------------------------

    @abstractmethod
    def _report(self, query: Query, series: list[Series]) -> None:
        """Render one resolved query with the concrete backend."""
        ...

    def report(self, metrics: MetricSchema) -> None:
        """Resolve and render every registered query against ``metrics``.

        ``metrics`` is not modified.
        """
        for query in self._queries:
            self._report(query, self._resolve_query(metrics, query))

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
        ...
