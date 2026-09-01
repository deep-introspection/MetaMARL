from abc import ABC, abstractmethod
from typing import TypeAlias
from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger
from enum import Enum

from core.metrics.metric.base import PrimitiveType
from core.metrics.schemas import MetricSchema
from core.reporting.query import Query

Path: TypeAlias = tuple[str | Enum, ...]
Group: TypeAlias = tuple[tuple[str, str], ...]
Resolved: TypeAlias = dict[Group, list[PrimitiveType]]


class Reporter(ABC):
    """Base interface for reporting reduced metric results.

    A Reporter receives populated MetricSchema objects, resolves its configured
    queries against those schemas, and delegates the resulting data to a
    backend-specific reporting implementation.

    Reporter views are write-once: they may be configured after construction,
    but cannot be replaced once set.
    """

    # TODO how to store data in the results reporter ?
    _queries: tuple[Query, ...] = ()
    _schema: type[MetricSchema] | None = None

    @property
    def queries(self) -> tuple[Query, ...]:
        """Return the queries registered with this reporter."""
        return self._queries

    def add_query(self, *queries: Query) -> None:
        """Register one or more reporting queries.

        Args:
            *queries: Queries to register with this reporter.
        """
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

    def _resolve_path(
        self,
        path: Path,
        metrics: MetricSchema | dict | list[PrimitiveType],
        *,
        index: int = 0,
        group: Group = (),
        junction: str | None = None,
    ) -> Resolved:
        """
        Returns the Metric object following the Path in a metric schema
        """
        if index >= len(path):
            if not isinstance(metrics, list):
                raise KeyError(f"Path does not point to a metric series: {path}")
            return {group: metrics}
        token = path[index]

        if isinstance(metrics, dict):
            if token == ReduceProtocol.SERIES:
                resolved: Resolved = {}
                for dynamic_id, child in sorted(
                    metrics.items(), key=lambda item: str(item[0])
                ):
                    child_group = group + (
                        (
                            junction or "dict",
                            str(dynamic_id),
                        ),
                    )
                    child_resolved = self._resolve_path(
                        path=path,
                        metrics=child,
                        index=index + 1,
                        group=child_group,
                    )
                    resolved.update(child_resolved)
                return resolved

            if token == ReduceProtocol.MEAN:
                branches = [
                    self._resolve_path(
                        path=path,
                        metrics=child,
                        index=index + 1,
                        group=group,
                    )
                    for child in metrics.values()
                ]
                if not branches:
                    return {}

                groups = set(branches[0])

                if any(set(branch) != groups for branch in branches[1:]):
                    raise ValueError(
                        "Cannot compute mean across branches with different SERIES groups."
                    )
                reduced: Resolved = {}
                for branch_group in groups:
                    series = [branch[branch_group] for branch in branches]
                    lengths = {len(values) for values in series}
                    if len(lengths) != 1:
                        raise ValueError(
                            "Cannot compute pointwise mean over series with different lengths: "
                            f"{sorted(lengths)}."
                        )
                    reduced[branch_group] = [
                        sum(float(value) for value in values) / len(values)
                        for values in zip(*series)
                    ]
                return reduced

            if isinstance(token, ReduceProtocol):
                raise NotImplementedError(
                    f"Dictionary query reduction {token} is not supported."
                )

            key = token.value if isinstance(token, Enum) else token
            try:
                child = metrics[key]
            except KeyError:
                raise KeyError(f"Unknown metric path: {path}") from None
            return self._resolve_path(
                path=path,
                metrics=child,
                index=index + 1,
                group=group,
            )

        if isinstance(token, ReduceProtocol):
            raise TypeError(
                f"{token} can only follow a dictionary field in path {path}."
            )

        key = token.value if isinstance(token, Enum) else token
        try:
            child = getattr(metrics, key)
        except AttributeError:
            raise KeyError(f"Unknown metric path: {path}") from None
        return self._resolve_path(
            path=path,
            metrics=child,
            index=index + 1,
            group=group,
            junction=key if isinstance(child, dict) else None,
        )

    def _resolve_query(
        self,
        metrics: MetricSchema,
        query: Query,
    ) -> tuple[Resolved, list[Resolved]]:
        """Resolve a query against a populated metric schema.

        Args:
            metrics: Reduced metric schema containing the values available for
                reporting.
            query: Query describing the metric paths to resolve.

        Returns:
            The resolved x and y values for the query.

        Raises:
            KeyError: If a requested metric path does not exist in the schema.
        """
        # Must return x and y series of same length
        xs = self._resolve_path(path=query.x, metrics=metrics)
        yss = [self._resolve_path(path=path, metrics=metrics) for path in query.y_paths]

        for path, ys in zip(query.y_paths, yss):
            if set(xs) == {()}:
                x = xs[()]

                for group, y in ys.items():
                    if len(x) != len(y):
                        raise ValueError(
                            "Query series must have equal length: "
                            f"x={query.x} ({len(x)}), y={path}, group={group} ({len(y)})."
                        )
                continue

            if set(xs) != set(ys):
                raise ValueError(
                    f"Dynamic x and y groups do not match: x={set(xs)}, y={set(ys)}."
                )

            for group in xs:
                x = xs[group]
                y = ys[group]

                if len(x) != len(y):
                    raise ValueError(
                        "Query series must have equal "
                        f"length for group {group}: x={len(x)}, y={len(y)}."
                    )
        return xs, yss

    @abstractmethod
    def _report(
        self,
        query: Query,
        x: Resolved,
        ys: list[Resolved],
    ) -> None:
        """Report one resolved query using the concrete reporting backend.

        Args:
            query: Query defining how the resolved values should be represented.
            x: Resolved values for the query's x dimension.
            y: Resolved values for the query's y dimension.
        """
        ...

    def report(self, metrics: MetricSchema) -> None:
        """Report all applicable configured views for a metric schema.

        Each configured query is resolved against `metrics`. Queries whose
        required values are available are forwarded to the backend-specific
        reporting implementation.

        Args:
            metrics: Reduced metric schema to report.
        """
        for query in self._queries:
            x, ys = self._resolve_query(metrics, query)
            self._report(query, x, ys)

    @abstractmethod
    def close(self) -> None:
        """Close the reporter instance"""
        ...
