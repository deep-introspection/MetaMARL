"""``Query`` validation and ``Reporter`` path/query resolution (TODO §8, §9)."""

import pytest

from core.metrics.logger import MetricLogger
from core.reporting.base import Reporter
from core.reporting.query import Query
from tests.metrics.conftest import LeafSchema, RootSchema


class RecordingReporter(Reporter):
    def __init__(self):
        self.reports = []

    def _report(self, query, x, ys):
        self.reports.append((query, x, ys))

    def close(self):
        pass


def populated_root():
    logger = MetricLogger.from_schema(RootSchema)
    for i in range(3):
        logger.push(("iter",), i)
        logger.push(("static", "series_value"), float(i * 2))
        logger.push(("static", "mean_value"), float(i))
        logger.push(("group", "by_id", "s1", "series_value"), 1.0 + i)
        logger.push(("group", "by_id", "s2", "series_value"), 3.0 + i)
    return logger.peek()


@pytest.mark.unit
class TestQuery:
    def test_single_and_multiple_y_paths(self):
        q = Query(title="t", x=("iter",), y=("a", "b"))
        assert q.y_paths == (("a", "b"),)
        q2 = Query(title="t", x=("iter",), y=(("a",), ("b", "c")))
        assert q2.y_paths == (("a",), ("b", "c"))

    def test_error_requires_reduction(self):
        with pytest.raises(ValueError, match="requires a reduction"):
            Query(title="t", x=("iter",), y=("a",), error="std")
        Query(title="t", x=("iter",), y=("a",), reduce="mean", error="std")

    def test_is_frozen(self):
        q = Query(title="t", x=("iter",), y=("a",))
        with pytest.raises(AttributeError):
            q.title = "u"


@pytest.mark.unit
class TestReporter:
    def test_schema_is_write_once_and_queries_accumulate(self):
        r = RecordingReporter()
        r.schema = RootSchema
        with pytest.raises(AttributeError):
            r.schema = LeafSchema
        q = Query(title="t", x=("iter",), y=("static", "series_value"))
        r.add_query(q)
        r.add_query(q, q)
        assert r.queries == (q, q, q)

    def test_resolve_static_nested_and_dynamic_paths(self):
        metrics = populated_root()
        r = RecordingReporter()
        assert r._resolve_path(("iter",), metrics) == [0, 1, 2]
        assert r._resolve_path(("static", "series_value"), metrics) == [0.0, 2.0, 4.0]
        assert r._resolve_path(("group", "by_id", "s2", "series_value"), metrics) == [
            3.0,
            4.0,
            5.0,
        ]

    def test_resolve_errors_name_the_path(self):
        metrics = populated_root()
        r = RecordingReporter()
        with pytest.raises(KeyError, match="static.*nope"):
            r._resolve_path(("static", "nope"), metrics)
        with pytest.raises(KeyError, match="does not point to a metric"):
            r._resolve_path((), metrics)
        with pytest.raises(TypeError, match="metric series"):
            r._resolve_path(("static",), metrics)
        with pytest.raises(KeyError, match="beyond metric leaf"):
            r._resolve_path(("static", "series_value", "x"), metrics)
        with pytest.raises(KeyError, match="Unknown metric path"):
            r._resolve_path(("group", "by_id", "zz", "series_value"), metrics)

    def test_resolve_query_checks_lengths(self):
        metrics = populated_root()
        r = RecordingReporter()
        q = Query(
            title="t",
            x=("iter",),
            y=(("static", "series_value"), ("static", "mean_value")),
        )
        x, ys = r._resolve_query(metrics, q)
        assert x == [0, 1, 2] and ys == [[0.0, 2.0, 4.0], [0.0, 1.0, 2.0]]

        short = MetricLogger.from_schema(RootSchema)
        short.push(("iter",), 0)
        short.push(("iter",), 1)
        short.push(("static", "series_value"), 1.0)
        with pytest.raises(ValueError, match="equal length"):
            r._resolve_query(
                short.peek(),
                Query(title="t", x=("iter",), y=("static", "series_value")),
            )

    def test_report_dispatches_every_query_without_mutating_metrics(self):
        metrics = populated_root()
        before = metrics.model_dump()
        r = RecordingReporter()
        r.add_query(
            Query(title="a", x=("iter",), y=("static", "series_value")),
            Query(title="b", x=("iter",), y=("group", "by_id", "s1", "series_value")),
        )
        r.report(metrics)
        assert [q.title for q, *_ in r.reports] == ["a", "b"]
        assert r.reports[1][2] == [[1.0, 2.0, 3.0]]
        assert metrics.model_dump() == before
        RecordingReporter().report(metrics)  # empty query list is a no-op
