"""Metric reducing to the sum of numeric values (``SUM``, empty -> 0)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class SumMetric(SeriesMetric):
    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"SumMetric only accepts int or float, got {type(value).__name__}."
            )
        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType]:
        if not compile:
            return list(self.values)
        return sum(self.values)

    # TODO move to base cls
    def reduce(
        self,
        compile: bool = True,
    ) -> PrimitiveType | SumMetric:
        sum = self.peek(compile=True)
        self.flush()
        if compile:
            return sum

        metric = SumMetric()
        metric.values = [sum]
        return metric

    def __repr__(self) -> str:
        return f"SumMetric({self.peek()}; len={len(self)})"
