"""Metric reducing to the minimum of numeric values (``MIN``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MinMetric(SeriesMetric):
    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MinMetric only accepts int or float, got {type(value).__name__}."
            )
        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType] | None:
        if not compile:
            return list(self.values)
        if not self.values:
            return None
        return min(self.values)

    # TODO move to base cls
    def reduce(
        self,
        compile: bool = True,
    ) -> PrimitiveType | MinMetric | None:
        if not self.values:
            return None if compile else MinMetric()
        min = self.peek(compile=True)
        self.flush()
        if compile:
            return min

        metric = MinMetric()
        metric.values = [min]
        return metric

    def __repr__(self) -> str:
        return f"MinMetric({self.peek()}; len={len(self)})"
