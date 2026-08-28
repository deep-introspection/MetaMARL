"""Metric reducing to the last pushed value (``LAST``)."""

from __future__ import annotations

from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class LastMetric(SeriesMetric):
    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType]:
        if not compile:
            return list(self.values)
        if not self.values:
            return None
        return self.values[-1]

    # TODO move to base cls
    def reduce(
        self,
        compile: bool = True,
    ) -> float | LastMetric:
        if not self.values:
            return None if compile else LastMetric()
        last = self.peek(compile=True)
        self.flush()
        if compile:
            return last

        metric = LastMetric()
        metric.values = [last]
        return metric

    def __repr__(self) -> str:
        return f"LastMetric({self.peek()}; len={len(self)})"
