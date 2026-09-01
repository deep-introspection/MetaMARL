from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MaxMetric(SeriesMetric):
    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MaxMetric only accepts int or float, got {type(value).__name__}."
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
        return max(self.values)

    def reduce(
        self,
        compile: bool = True,
    ) -> PrimitiveType | MaxMetric | None:
        if not self.values:
            return None if compile else MaxMetric()
        max = self.peek(compile=True)
        self.flush()
        if compile:
            return max

        metric = MaxMetric()
        metric.values = [max]
        return metric

    def __repr__(self) -> str:
        return f"MaxMetric({self.peek()}; len={len(self)})"
