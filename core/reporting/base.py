from abc import ABC, abstractmethod
from typing import Optional

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
    
    _views: tuple[Query, ...]

    @property
    def views(self):
        if self._views is None:
            raise RuntimeError("Reporter views have not been set.")
        return self._views

    @views.setter
    def set_views(self, views: list[Query]) -> None:
        if self._views is not None:
            raise RuntimeError("Reporter views have already been set.")
        self._views = tuple(views)


    @abstractmethod
    def _resolve_query(
        self,
        metrics: MetricSchema,
        query: Query,
    ):
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
        ...

    @abstractmethod
    def _report_query(
        self,
        query: Query,
        x,
        y,
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
        for query in self._views:
            x, y = self._resolve_query(metrics, query)
            self._report_query(query, x, y)

    @abstractmethod
    def export(self, metrics: MetricSchema) -> None:
        """Export a populated metric schema using the concrete backend.

        This method handles backend-specific persistence or serialization of
        metric results independently of configured views.

        Args:
            metrics: Reduced metric schema to export.
        """
        ...