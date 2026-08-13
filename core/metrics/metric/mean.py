from __future__ import annotations
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MeanMetric(SeriesMetric):

    def push(self, value: PrimitiveType) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MeanMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> float | list[float]:
        if not compile :
            return list(self.values)
        
        if not self.values:
            return float("nan") if compile else []

        return sum(self.values) / len(self.values)

    def reduce(
        self,
        compile: bool = True,
    ) -> float | MeanMetric:
        mean = self.peek(compile=True)
        self.flush()

        if compile:
            return mean

        metric = MeanMetric()
        metric.values = [mean]
        return metric

    def __repr__(self) -> str:
        return f"MeanMetric({self.peek()}; len={len(self)})"