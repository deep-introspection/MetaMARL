from abc import ABC, abstractmethod

from core.metrics.logger import Path
from core.metrics.metric.base import Metric
from core.metrics.metric.series import SeriesMetric
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
    def schema(self, schema:  type[MetricSchema]) -> None:
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
            index: int = 0
        ) -> SeriesMetric:
        """
        Returns the Metric object following the Path in a metric schema
        """
        if index >= len(path): raise KeyError(f"Path does not point to a metric: {path}")
        key = path[index]

        if isinstance(metrics, dict):
            try:
                child = metrics[key]
            except KeyError: raise KeyError(f"Unknown metric path: {path}") from None
        
        else:
            try:
                child = getattr(metrics, key)
            except AttributeError: raise KeyError(f"Unknown metric path: {path}") from None

        if isinstance(child, Metric):
            if index != len(path) - 1: raise KeyError(f"Path continues beyond metric leaf: {path}")
            if not isinstance(child, SeriesMetric): 
                raise TypeError(f"Path does not point to a SeriesMetric: {path}")
            return child
        return self._resolve_path(path=path, metrics=child, index=index + 1)

    def _resolve_query(
        self,
        metrics: MetricSchema,
        query: Query,
    ) -> tuple[SeriesMetric, SeriesMetric]:
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
        x_series = self._resolve_path(path=query.x, metrics=metrics)
        y_series = self._resolve_path(path=query.y, metrics=metrics)
        return x_series, y_series


    @abstractmethod
    def _report(
        self,
        query: Query,
        x: SeriesMetric,
        y: SeriesMetric,
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
            x, y = self._resolve_query(metrics, query)
            self._report(query, x, y)

    @abstractmethod
    def close(self) -> None:
        """Close the reporter instance
        """
        ...