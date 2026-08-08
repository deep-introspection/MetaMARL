from core.metrics.metric.base import Metric, PrimitiveType


class SeriesMetric(Metric):

    def __init__(self) -> None:
        self.values: list[PrimitiveType] = []

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
            return f"SeriesMetric(len={len(self)})"

    def push(self, value: PrimitiveType) -> None:
        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> list[PrimitiveType]:
        return list(self.values)

    def reduce(
        self,
        compile: bool = True,
    ) -> list[PrimitiveType] | "SeriesMetric":
        values = list(self.values)
        self.flush()

        if compile:
            return values

        metric = SeriesMetric()
        metric.values = values
        return metric

    def flush(self) -> None:
        self.values.clear()