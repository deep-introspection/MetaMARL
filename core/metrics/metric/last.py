from __future__ import annotations
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class LastMetric(SeriesMetric):
    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType]:
        if not self.values:
            return float("nan") if compile else []

        last = self.values[-1]

        if compile:
            return last

        return [last]

    def reduce(
        self,
        compile: bool = True,
    ) -> float | LastMetric:
        if not self.values:
            last = float("nan")
        else:
            last = self.values[-1]

        self.flush()

        if compile:
            return last

        metric = LastMetric()
        metric.values = [last]
        return metric

    def __repr__(self) -> str:
        return f"LastMetric({self.peek()}; len={len(self)})"