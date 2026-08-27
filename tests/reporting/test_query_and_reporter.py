"""``Query`` validation and ``Reporter`` path/query resolution (TODO §8, §9)."""

import pytest

from core.metrics.logger import MetricLogger
from core.reporting.base import Reporter
from core.reporting.query import Query
from tests.metrics.conftest import LeafSchema, RootSchema


class RecordingReporter(Reporter):
    def __init__(self):
        self.reports = []

    def _report(self, query, series):
        self.reports.append((query, series))

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
        series = r._resolve_query(metrics, q)
        assert [s.label for s in series] == ["static/series_value", "static/mean_value"]
        assert series[0].x == [0, 1, 2] and series[0].y == [0.0, 2.0, 4.0]
        assert series[1].y == [0.0, 1.0, 2.0] and series[1].error is None

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
        assert [q.title for q, _ in r.reports] == ["a", "b"]
        assert r.reports[1][1][0].y == [1.0, 2.0, 3.0]
        assert metrics.model_dump() == before
        RecordingReporter().report(metrics)  # empty query list is a no-op

    def test_mean_without_wildcards_averages_the_listed_paths(self):
        metrics = populated_root()
        r = RecordingReporter()
        q = Query(
            title="avg",
            x=("iter",),
            y=(
                ("group", "by_id", "s1", "series_value"),
                ("group", "by_id", "s2", "series_value"),
            ),
            reduce="mean",
            error="std",
        )
        (s,) = r._resolve_query(metrics, q)
        assert (
            s.label == "avg" and s.y == [2.0, 3.0, 4.0] and s.error == [1.0, 1.0, 1.0]
        )


def populated_nested():
    """mechanism -> seed -> value, with values chosen so a binding error cannot pass."""
    from typing import Optional

    from pydantic import Field

    from core.metrics.enums import ReduceProtocol
    from core.metrics.schemas import MetricSchema

    class SeedS(MetricSchema):
        value: Optional[float] = Field(
            default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
        )

    class MechS(MetricSchema):
        by_seed: dict[str, SeedS] = Field(default_factory=dict)
        fitness: Optional[float] = Field(
            default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
        )
        param: Optional[float] = Field(
            default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
        )

    class RootS(MetricSchema):
        by_mechanism: dict[str, MechS] = Field(default_factory=dict)

    logger = MetricLogger.from_schema(RootS)
    for it in range(3):
        logger.push(("iter",), it)
        for m, base in (("m0", 10.0), ("m1", 20.0)):
            logger.push(("by_mechanism", m, "fitness"), base + it)
            logger.push(("by_mechanism", m, "param"), base / 100 + it)
            for seed, offset in (("s1", 1.0), ("s2", 3.0)):
                logger.push(
                    ("by_mechanism", m, "by_seed", seed, "value"), base + offset + it
                )
    return logger.peek()


@pytest.mark.unit
class TestWildcards:
    def test_one_wildcard_expands_sorted_with_bound_labels(self):
        r = RecordingReporter()
        q = Query(title="t", x=("iter",), y=("by_mechanism", "*", "fitness"))
        series = r._resolve_query(populated_nested(), q)
        assert [s.label for s in series] == [
            "by_mechanism/m0/fitness",
            "by_mechanism/m1/fitness",
        ]
        assert series[0].y == [10.0, 11.0, 12.0] and series[1].y == [20.0, 21.0, 22.0]

    def test_two_wildcards_and_concrete_plus_wildcard(self):
        r = RecordingReporter()
        series = r._resolve_query(
            populated_nested(),
            Query(
                title="t", x=("iter",), y=("by_mechanism", "*", "by_seed", "*", "value")
            ),
        )
        assert [s.label for s in series] == [
            "by_mechanism/m0/by_seed/s1/value",
            "by_mechanism/m0/by_seed/s2/value",
            "by_mechanism/m1/by_seed/s1/value",
            "by_mechanism/m1/by_seed/s2/value",
        ]
        series = r._resolve_query(
            populated_nested(),
            Query(
                title="t",
                x=("iter",),
                y=("by_mechanism", "m1", "by_seed", "*", "value"),
            ),
        )
        assert [s.label for s in series] == [
            "by_mechanism/m1/by_seed/s1/value",
            "by_mechanism/m1/by_seed/s2/value",
        ]
        assert series[1].y == [23.0, 24.0, 25.0]

    def test_wildcard_only_at_dynamic_nodes_and_no_match(self):
        r = RecordingReporter()
        with pytest.raises(KeyError, match="only valid at a dynamic node"):
            r._resolve_query(
                populated_nested(), Query(title="t", x=("iter",), y=("*", "fitness"))
            )
        with pytest.raises(KeyError, match="does not expand wildcards"):
            r._resolve_path(("by_mechanism", "*", "fitness"), populated_nested())
        empty = MetricLogger.from_schema(type(populated_nested())).peek()
        assert (
            r._resolve_query(
                empty, Query(title="t", x=("iter",), y=("by_mechanism", "*", "fitness"))
            )
            == []
        )

    def test_grouping_reduces_across_seeds_per_mechanism(self):
        r = RecordingReporter()
        q = Query(
            title="t",
            x=("iter",),
            y=("by_mechanism", "*", "by_seed", "*", "value"),
            reduce="mean",
            error="std",
        )
        series = r._resolve_query(populated_nested(), q)
        assert [s.label for s in series] == ["m0", "m1"]  # one group per mechanism
        assert series[0].y == [12.0, 13.0, 14.0]  # mean over seeds only (11, 13) -> 12
        assert series[1].y == [22.0, 23.0, 24.0]
        assert series[0].error == [1.0, 1.0, 1.0]

    def test_single_wildcard_mean_averages_across_matches(self):
        r = RecordingReporter()
        q = Query(
            title="avg",
            x=("iter",),
            y=("by_mechanism", "*", "fitness"),
            reduce="mean",
            error="std",
        )
        (s,) = r._resolve_query(populated_nested(), q)
        assert s.label == "avg"
        assert s.y == [15.0, 16.0, 17.0]  # mean of (10, 20) + it
        assert s.error == [5.0, 5.0, 5.0]

    def test_wildcard_binding_aligns_x_and_y(self):
        r = RecordingReporter()
        q = Query(
            title="scatter",
            x=("by_mechanism", "*", "param"),
            y=("by_mechanism", "*", "fitness"),
        )
        series = r._resolve_query(populated_nested(), q)
        assert [(s.x, s.y) for s in series] == [
            ([0.1, 1.1, 2.1], [10.0, 11.0, 12.0]),
            ([0.2, 1.2, 2.2], [20.0, 21.0, 22.0]),
        ]

    def test_x_without_matching_binding_is_an_error(self):
        r = RecordingReporter()
        q = Query(
            title="t",
            x=("by_mechanism", "m0", "by_seed", "*", "value"),
            y=("by_mechanism", "*", "fitness"),
        )
        with pytest.raises(KeyError, match="No x series for bindings"):
            r._resolve_query(populated_nested(), q)
