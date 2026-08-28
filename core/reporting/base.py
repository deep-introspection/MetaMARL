"""Reporter base: resolve queries against a populated metric schema.

``Reporter.report(metrics)`` resolves every registered :class:`Query` into a
list of :class:`Series` — expanding wildcards, aligning x and y by their
wildcard bindings and applying the requested reduction — then hands them to
the backend-specific :meth:`Reporter._report`. Backends therefore never see
raw metric trees, only labeled ``(x, y[, error][, color])`` series.

A :class:`ParallelCoordinatesQuery` resolves through :meth:`Reporter._resolve_parallel`
into a :class:`Table` (one row per evaluated entity and index) handed to
:meth:`Reporter._report_table`; backends that cannot draw parallel coordinates
inherit a default that logs a warning and skips the query.
"""

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Union, get_args, get_origin

import numpy as np

from core.metrics.metric.base import PrimitiveType
from core.metrics.schemas import MetricSchema
from core.reporting.query import (
    WILDCARD,
    AnyQuery,
    ParallelCoordinatesQuery,
    Path,
    Query,
    Series,
    Table,
)

logger = logging.getLogger(__name__)

Bindings = tuple[str, ...]
"""Runtime ids bound to the wildcards of a path, in path order."""

Entity = str | None
"""Row key of a parallel-coordinates expansion: the first wildcard binding, or
``None`` for a series shared by every entity (a root ``generation`` series)."""


class Reporter(ABC):
    """Base interface for reporting resolved metric series.

    Reporters are configured after construction (``schema`` is write-once,
    queries accumulate) and are driven by :meth:`report`.
    """

    _queries: tuple[AnyQuery, ...] = ()
    _schema: type[MetricSchema] | None = None

    # --- configuration -------------------------------------------------------------

    @property
    def queries(self) -> tuple[AnyQuery, ...]:
        """Queries registered with this reporter."""
        return self._queries

    def add_query(self, *queries: AnyQuery) -> None:
        """Register one or more reporting queries (line/scatter or parallel coordinates)."""
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

    def _resolve_axis(
        self, path: Path, metrics: MetricSchema
    ) -> dict[Bindings, list[PrimitiveType]]:
        """Expand an auxiliary axis (x or color) into its series keyed by bindings."""
        return {
            bindings: values for bindings, values in self._expand_path(path, metrics)
        }

    @staticmethod
    def _axis_for(
        axis_by_binding: dict[Bindings, list[PrimitiveType]],
        y_bindings: Bindings,
        query: Query,
        path: Path,
        name: str = "x",
    ) -> list[PrimitiveType]:
        """Pick the axis series sharing the y series' leading bindings (no Cartesian product)."""
        if len(axis_by_binding) == 1 and () in axis_by_binding:
            return axis_by_binding[()]
        for axis_bindings, values in axis_by_binding.items():
            if y_bindings[: len(axis_bindings)] == axis_bindings:
                return values
        raise KeyError(
            f"No {name} series for bindings {y_bindings} in query {query.title!r}: "
            f"{name}={path} binds {sorted(axis_by_binding)}"
        )

    def _aligned_axis(
        self,
        axis_by_binding: dict[Bindings, list[PrimitiveType]],
        bindings: Bindings,
        y_label: str,
        y_length: int,
        query: Query,
        path: Path,
        name: str,
    ) -> list[PrimitiveType]:
        """``_axis_for`` plus the length check against the y series."""
        values = self._axis_for(axis_by_binding, bindings, query, path, name)
        if len(values) != y_length:
            raise ValueError(
                f"Query series must have equal length: {name}={path} "
                f"({len(values)}), y={y_label} ({y_length})."
            )
        return values

    def _resolve_query(self, metrics: MetricSchema, query: Query) -> list[Series]:
        """Resolve ``query`` against ``metrics`` into labeled series.

        Raises
        ------
        KeyError
            If a path does not exist.
        ValueError
            If x and y lengths differ, or replicates of a group differ in length.
        """
        x_by_binding = self._resolve_axis(query.x, metrics)
        color_by_binding = (
            self._resolve_axis(query.color, metrics)
            if query.color is not None
            else None
        )

        raw: list[tuple[Bindings, str, list, list, list | None]] = []
        for path in query.y_paths:
            for bindings, values in self._expand_path(path, metrics):
                label = self._label(path, bindings)
                x = self._aligned_axis(
                    x_by_binding, bindings, label, len(values), query, query.x, "x"
                )
                color = None
                if color_by_binding is not None and query.color is not None:
                    color = self._aligned_axis(
                        color_by_binding,
                        bindings,
                        label,
                        len(values),
                        query,
                        query.color,
                        "color",
                    )
                raw.append((bindings, label, x, values, color))

        if query.reduce == "none":
            return [
                Series(
                    label=label,
                    x=list(x),
                    y=list(y),
                    color=list(color) if color is not None else None,
                )
                for _, label, x, y, color in raw
            ]

        if query.reduce != "mean":
            raise ValueError(f"Unknown query reduction: {query.reduce!r}")

        # Group by the first wildcard binding (e.g. one group per mechanism) and
        # average over the remaining ones (e.g. seeds). Without wildcards, the
        # listed y paths form a single group labeled by the query title.
        # A single wildcard level has nothing left to average once grouped, so
        # its matches are the replicates of one group (mean across candidates,
        # agents, ...). Two or more levels group by the first one.
        groups: "OrderedDict[str, list[tuple[list, list]]]" = OrderedDict()
        for bindings, _, x, y, _ in raw:
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

    # --- parallel-coordinates resolution ---------------------------------------------

    @staticmethod
    def _unwrap_optional(annotation: Any) -> Any:
        """``Optional[X]`` -> ``X`` (pydantic field annotations)."""
        if get_origin(annotation) is Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return annotation

    @classmethod
    def _dynamic_positions(cls, path: Path, metrics: MetricSchema | dict) -> list[int]:
        """Indices of the path segments that are keys of a dynamic node.

        The walk follows the first key of every populated dynamic node (all
        values of one node share a schema) and continues on the field
        annotation (``dict[ID, Schema]``) from the first empty node, so the
        column labels of a query are known before any entity was logged.
        """
        positions: list[int] = []
        node: Any = metrics
        for index, key in enumerate(path):
            if isinstance(node, dict):
                positions.append(index)
                if key == WILDCARD and node:
                    node = node[sorted(node, key=str)[0]]
                elif key in node:
                    node = node[key]
                else:
                    break  # unknown id: nothing expands there either
            elif isinstance(node, MetricSchema):
                child = getattr(node, key, None)
                if isinstance(child, dict) and not child:
                    annotation = type(node).model_fields[key].annotation
                    positions.extend(
                        cls._static_positions(annotation, path[index + 1 :], index + 1)
                    )
                    break
                node = child
            else:
                break
        return positions

    @classmethod
    def _static_positions(cls, annotation: Any, path: Path, offset: int) -> list[int]:
        """Dynamic positions of ``path`` (starting at ``offset``) read from annotations."""
        positions: list[int] = []
        node: Any = cls._unwrap_optional(annotation)
        for index, key in enumerate(path):
            if isinstance(node, type) and issubclass(node, MetricSchema):
                if key not in node.model_fields:
                    break
                node = cls._unwrap_optional(node.model_fields[key].annotation)
                continue
            if get_origin(node) is dict:
                positions.append(offset + index)
                node = cls._unwrap_optional(get_args(node)[1])
                continue
            break
        return positions

    @classmethod
    def _axis_label(
        cls, path: Path, bindings: Bindings, dynamic: list[int]
    ) -> str | None:
        """Axis label of one expansion (see :meth:`_resolve_parallel`).

        ``None`` when the label is a wildcard id that has no binding yet.
        """
        if WILDCARD not in path:
            return path[-1]
        entity_index = path.index(WILDCARD)
        after_entity = [p for p in dynamic if p > entity_index]
        if not after_entity:
            return path[-1]
        position = after_entity[-1]
        if path[position] != WILDCARD:
            return path[position]
        wildcard_rank = path[: position + 1].count(WILDCARD) - 1
        return bindings[wildcard_rank] if wildcard_rank < len(bindings) else None

    def _expand_axis(
        self, path: Path, metrics: MetricSchema
    ) -> "OrderedDict[str, dict[Entity, list[PrimitiveType]]]":
        """Expand one parallel-coordinates path into ``{label: {entity: series}}``."""
        dynamic = self._dynamic_positions(path, metrics)
        axes: "OrderedDict[str, dict[Entity, list[PrimitiveType]]]" = OrderedDict()
        expansions = self._expand_path(path, metrics)
        if not expansions:
            # No entity yet: still declare the column when its label is static.
            label = self._axis_label(path, (), dynamic)
            if label is not None:
                axes[label] = {}
            return axes
        for bindings, values in expansions:
            entity: Entity = bindings[0] if bindings else None
            label = self._axis_label(path, bindings, dynamic)
            assert label is not None  # bindings are complete for a real expansion
            axes.setdefault(label, {})[entity] = values
        return axes

    def _resolve_parallel(
        self, metrics: MetricSchema, query: ParallelCoordinatesQuery
    ) -> Table:
        """Resolve ``query`` into one :class:`Table`.

        Rules
        -----
        - Every dimension path is expanded like a query path. The *entity*
          (row key) of an expansion is its first wildcard binding; an expansion
          without wildcard is shared by every entity (a root ``generation``).
          This is the mean-grouping convention: the first wildcard groups.
        - The *axis label* of an expansion is the last dynamic key bound after
          the entity wildcard (``"fixed_quota"`` in
          ``by_mechanism/*/by_parameter/fixed_quota/value``, or the id bound by
          a second wildcard), and the leaf field name otherwise (``"fitness"``).
          The same label produced by two different paths is an error.
        - For each entity (sorted by ``str``) every column and the colour must
          have the same length ``L``. Rows are emitted index-major then entity,
          so the table accumulates generation after generation.

        Raises
        ------
        ValueError
            Duplicate axis label, a column missing for an entity, or series of
            different lengths for one entity.
        """
        columns: "OrderedDict[str, dict[Entity, list[PrimitiveType]]]" = OrderedDict()
        for path in query.dimensions:
            for label, by_entity in self._expand_axis(path, metrics).items():
                if label in columns:
                    raise ValueError(
                        f"Duplicate axis label {label!r} in parallel-coordinates "
                        f"query {query.title!r} (path {path})."
                    )
                columns[label] = by_entity
        color_axes = self._expand_axis(query.color, metrics)
        if len(color_axes) > 1:
            raise ValueError(
                f"Color path {query.color} of query {query.title!r} expands to "
                f"several axes: {list(color_axes)}."
            )
        color_label = next(iter(color_axes), query.color[-1])
        color_by_entity = color_axes.get(color_label, {})

        entities: set[Entity] = set()
        for by_entity in (*columns.values(), color_by_entity):
            entities.update(e for e in by_entity if e is not None)
        if not entities:
            shared = (*columns.values(), color_by_entity)
            if all(None in by_entity for by_entity in shared):
                entities.add(None)  # everything is shared: one anonymous entity
            else:
                # Nothing evaluated yet (empty dynamic node): keep the known columns.
                return Table(
                    columns=tuple(columns), rows=[], color=[], color_label=color_label
                )
        ordered = sorted(entities, key=str)

        def pick(by_entity: dict[Entity, list], entity: Entity, name: str) -> list:
            values = by_entity.get(entity, by_entity.get(None))
            if values is None:
                raise ValueError(
                    f"Entity {entity!r} has no column {name!r} in "
                    f"parallel-coordinates query {query.title!r}."
                )
            return values

        rows: list[list[float]] = []
        color: list[float] = []
        per_entity: list[tuple[list[list], list]] = []
        for entity in ordered:
            cols = [
                pick(by_entity, entity, name) for name, by_entity in columns.items()
            ]
            col_color = pick(color_by_entity, entity, color_label)
            lengths = {len(c) for c in cols} | {len(col_color)}
            if len(lengths) != 1:
                raise ValueError(
                    f"Series of entity {entity!r} in parallel-coordinates query "
                    f"{query.title!r} are not aligned: lengths {sorted(lengths)}."
                )
            per_entity.append((cols, col_color))
        length = len(per_entity[0][1]) if per_entity else 0
        for k in range(length):
            for cols, col_color in per_entity:
                rows.append([c[k] for c in cols])
                color.append(col_color[k])
        return Table(
            columns=tuple(columns), rows=rows, color=color, color_label=color_label
        )

    # --- backend contract --------------------------------------------------------------

    @abstractmethod
    def _report(self, query: Query, series: list[Series]) -> None:
        """Render one resolved query with the concrete backend."""
        ...

    def _report_table(self, query: ParallelCoordinatesQuery, table: Table) -> None:
        """Render one resolved parallel-coordinates query.

        The default skips the query with a warning; backends that can draw
        parallel coordinates (W&B, CSV) override it.
        """
        logger.warning(
            "%s does not render parallel coordinates; skipping query %r",
            type(self).__name__,
            query.title,
        )

    def report(self, metrics: MetricSchema) -> None:
        """Resolve and render every registered query against ``metrics``.

        ``metrics`` is not modified.
        """
        for query in self._queries:
            if isinstance(query, ParallelCoordinatesQuery):
                self._report_table(query, self._resolve_parallel(metrics, query))
            else:
                self._report(query, self._resolve_query(metrics, query))

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""
        ...
