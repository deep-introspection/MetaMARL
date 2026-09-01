from abc import ABC, abstractmethod

from core.metrics.logger import Path
from core.metrics.metric.base import PrimitiveType
from core.metrics.schemas import MetricSchema
from core.reporting.query import Query


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
        metrics: MetricSchema | dict[str, MetricSchema],
        *,
        index: int = 0,
    ) -> list[PrimitiveType]:
        """
        Returns the Metric object following the Path in a metric schema
        """
        if index >= len(path):
            raise KeyError(f"Path does not point to a metric: {path}")
        key = path[index]

        if isinstance(metrics, dict):
            try:
                child = metrics[key]
            except KeyError:
                raise KeyError(f"Unknown metric path: {path}") from None

        else:
            try:
                child = getattr(metrics, key)
            except AttributeError:
                raise KeyError(f"Unknown metric path: {path}") from None

        if index == len(path) - 1:
            if not isinstance(child, list):
                raise TypeError(f"Path does not point to a metric series: {path}")
            return child

        if not isinstance(child, (MetricSchema, dict)):
            raise KeyError(f"Path continues beyond metric leaf: {path}")
        return self._resolve_path(path=path, metrics=child, index=index + 1)

    def _resolve_query(
        self,
        metrics: MetricSchema,
        query: Query,
    ) -> tuple[list[PrimitiveType], list[list[PrimitiveType]]]:
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
        x = self._resolve_path(path=query.x, metrics=metrics)
        ys = [self._resolve_path(path=path, metrics=metrics) for path in query.y_paths]

        for path, y in zip(query.y_paths, ys):
            if len(x) != len(y):
                raise ValueError(
                    f"Query series must have equal length: "
                    f"x={query.x} ({len(x)}), "
                    f"y={path} ({len(y)})."
                )
        return x, ys

    @abstractmethod
    def _report(
        self,
        query: Query,
        x: list[PrimitiveType],
        ys: list[list[PrimitiveType]],
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
