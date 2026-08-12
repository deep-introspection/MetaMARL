from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import Metric
from core.metrics.metric.last import LastMetric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.series import SeriesMetric


class MetricFactory:

    @staticmethod
    def create(protocol: ReduceProtocol) -> Metric:
        match protocol:
            case ReduceProtocol.MEAN:
                return MeanMetric()

            case ReduceProtocol.SERIES:
                return SeriesMetric()

            case ReduceProtocol.LAST:
                return LastMetric()

            case _:
                raise NotImplementedError(
                    f"Reduce protocol {protocol.value!r} is not implemented."
                )