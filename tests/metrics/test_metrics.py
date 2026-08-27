"""Each ``Metric`` reducer: push/peek/reduce/flush semantics and empty values (TODO §7.7)."""

import pytest

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.last import LastMetric
from core.metrics.metric.max import MaxMetric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.min import MinMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.metric.sum import SumMetric

CASES = [
    (ReduceProtocol.MEAN, MeanMetric, [1.0, 2.0, 6.0], 3.0, None),
    (ReduceProtocol.SERIES, SeriesMetric, [1.0, 2.0, 6.0], [1.0, 2.0, 6.0], []),
    (ReduceProtocol.LAST, LastMetric, [1, 2, 6], 6, None),
    (ReduceProtocol.SUM, SumMetric, [1.0, 2.0, 6.0], 9.0, 0),
    (ReduceProtocol.MIN, MinMetric, [1.0, 2.0, 6.0], 1.0, None),
    (ReduceProtocol.MAX, MaxMetric, [1.0, 2.0, 6.0], 6.0, None),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "protocol, cls, values, reduced, empty", CASES, ids=[c[0].value for c in CASES]
)
def test_factory_push_peek_reduce(protocol, cls, values, reduced, empty):
    metric = MetricFactory.create(protocol)
    assert isinstance(metric, cls)
    assert metric.reduce() == empty  # empty reducer semantics

    for v in values:
        metric.push(v)
    assert len(metric) == len(values)
    assert metric.peek(compile=False) == values  # raw history
    assert metric.peek() == reduced  # non-destructive
    assert metric.peek() == reduced
    assert metric.reduce() == reduced  # destructive
    assert len(metric) == 0
    assert metric.reduce() == empty


@pytest.mark.unit
def test_reduce_without_compile_returns_metric_holding_the_value():
    m = MeanMetric()
    m.push(1.0), m.push(3.0)
    reduced = m.reduce(compile=False)
    assert isinstance(reduced, MeanMetric) and reduced.peek() == 2.0
    s = SeriesMetric()
    s.push(1.0)
    assert s.reduce(compile=False).peek() == [1.0]
    assert isinstance(MeanMetric().reduce(compile=False), MeanMetric)


@pytest.mark.unit
def test_numeric_metrics_reject_non_numbers():
    for m in (MeanMetric(), SumMetric(), MinMetric(), MaxMetric()):
        with pytest.raises(TypeError):
            m.push("x")
        with pytest.raises(TypeError):
            m.push(True)
    LastMetric().push("anything")  # LAST/SERIES accept any primitive


@pytest.mark.unit
def test_float_int_conversions_and_repr():
    m = MeanMetric()
    m.push(2), m.push(4)
    assert float(m) == 3.0 and int(m) == 3
    assert "MeanMetric" in repr(m)
    s = SeriesMetric()
    s.push(1.0)
    with pytest.raises(ValueError):
        float(s)
    assert isinstance(m.empty_copy(), MeanMetric) and len(m.empty_copy()) == 0


@pytest.mark.unit
def test_unimplemented_protocol():
    with pytest.raises(NotImplementedError):
        MetricFactory.create(ReduceProtocol.EMA)
