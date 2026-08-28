"""Per-point colour on ``Query`` and ``ParallelCoordinatesQuery`` (TODO §5.4, §5.5).

The synthetic schemas mirror ``ESSchema``: a root ``generation`` series shared
by every candidate, one ``fitness`` series per candidate under ``by_mechanism``
and one ``value`` series per optimized parameter under ``by_parameter``.
"""

import csv
import logging
from types import SimpleNamespace
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import pytest
from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.es.schema import ESSchema
from core.reporting.base import Reporter
from core.reporting.csv import CSVReporter
from core.reporting.query import ParallelCoordinatesQuery, Query, Series, Table
from core.reporting.tensor_board import TensorBoardReporter
from core.reporting.wandb import WandbConfig, WandbReporter
from core.utils import sanitize_key
from examples.bilevel_fishery.queries import (
    es_parallel_coordinates_query,
    es_parameter_fitness_queries,
)
from tests.optimizers.test_es_payload import InnerSchema, make_es


class _ParamS(MetricSchema):
    value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )


class _MechS(MetricSchema):
    fitness: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    gen: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    by_parameter: dict[str, _ParamS] = Field(default_factory=dict)


class _RootS(MetricSchema):
    generation: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    by_mechanism: dict[str, _MechS] = Field(default_factory=dict)


def _population(
    candidates: dict[str, dict[str, list[float]]],
    *,
    generations: int,
    fitness: dict[str, list[float]] | None = None,
):
    """Push ``generations`` records; ``candidates[id][param]`` lists one value per generation."""
    logger = MetricLogger.from_schema(_RootS)
    for gen in range(generations):
        logger.push(("generation",), gen)
        for cid, params in candidates.items():
            fit = fitness[cid][gen] if fitness else float(gen + int(cid[1:]))
            logger.push(("by_mechanism", cid, "fitness"), fit)
            logger.push(("by_mechanism", cid, "gen"), gen * 10 + int(cid[1:]))
            for name, values in params.items():
                logger.push(
                    ("by_mechanism", cid, "by_parameter", name, "value"), values[gen]
                )
    return logger.peek()


class RecordingReporter(Reporter):
    def __init__(self):
        self.reports = []

    def _report(self, query, series):
        self.reports.append((query, series))

    def close(self):
        pass


# --- validation -------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    def test_color_requires_no_reduction(self):
        with pytest.raises(ValueError, match="reduce='none'"):
            Query(
                title="t", x=("iter",), y=("a",), reduce="mean", color=("generation",)
            )

    def test_empty_color_is_rejected(self):
        with pytest.raises(ValueError, match="color path must not be empty"):
            Query(title="t", x=("iter",), y=("a",), color=())

    def test_color_is_accepted_without_reduction(self):
        q = Query(title="t", x=("iter",), y=("a",), color=("generation",))
        assert q.color == ("generation",)
        assert Series("l", [1], [2.0]).color is None

    def test_parallel_requires_a_dimension(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            ParallelCoordinatesQuery(title="t", dimensions=(), color=("f",))

    def test_parallel_rejects_empty_dimension_path(self):
        with pytest.raises(ValueError, match="dimension paths must not be empty"):
            ParallelCoordinatesQuery(title="t", dimensions=(("a",), ()), color=("f",))

    def test_parallel_rejects_empty_color(self):
        with pytest.raises(ValueError, match="color path must not be empty"):
            ParallelCoordinatesQuery(title="t", dimensions=(("a",),), color=())


# --- colour resolution ------------------------------------------------------------


@pytest.mark.unit
class TestColorResolution:
    def test_color_bound_to_the_same_entity_as_y(self):
        metrics = _population(
            {"m0": {"q": [0.1, 0.2]}, "m1": {"q": [0.3, 0.4]}}, generations=2
        )
        q = Query(
            title="t",
            x=("by_mechanism", "*", "by_parameter", "q", "value"),
            y=("by_mechanism", "*", "fitness"),
            color=("by_mechanism", "*", "gen"),
        )
        s0, s1 = RecordingReporter()._resolve_query(metrics, q)
        assert s0.label == "by_mechanism/m0/fitness" and s0.color == [0, 10]
        assert s1.label == "by_mechanism/m1/fitness" and s1.color == [1, 11]
        assert s0.x == [0.1, 0.2] and s1.x == [0.3, 0.4]

    def test_root_color_is_shared_by_every_candidate(self):
        metrics = _population(
            {"m0": {"q": [0.1, 0.2, 0.3]}, "m1": {"q": [0.3, 0.4, 0.5]}}, generations=3
        )
        q = Query(
            title="t",
            x=("by_mechanism", "*", "by_parameter", "q", "value"),
            y=("by_mechanism", "*", "fitness"),
            color=("generation",),
        )
        series = RecordingReporter()._resolve_query(metrics, q)
        assert [s.color for s in series] == [[0, 1, 2], [0, 1, 2]]

    def test_color_length_mismatch_is_an_error(self):
        metrics = _population({"m0": {"q": [0.1, 0.2]}}, generations=2)
        metrics.by_mechanism["m0"].gen.append(99)
        q = Query(
            title="t",
            x=("by_mechanism", "*", "by_parameter", "q", "value"),
            y=("by_mechanism", "*", "fitness"),
            color=("by_mechanism", "*", "gen"),
        )
        with pytest.raises(ValueError, match="equal length.*color="):
            RecordingReporter()._resolve_query(metrics, q)

    def test_series_without_color_keep_none(self):
        metrics = _population({"m0": {"q": [0.1, 0.2]}}, generations=2)
        q = Query(title="t", x=("generation",), y=("by_mechanism", "*", "fitness"))
        (s,) = RecordingReporter()._resolve_query(metrics, q)
        assert s.color is None


# --- parallel-coordinates resolution ----------------------------------------------


PARALLEL = ParallelCoordinatesQuery(
    title="pc",
    dimensions=(
        ("by_mechanism", "*", "by_parameter", "*", "value"),
        ("by_mechanism", "*", "fitness"),
    ),
    color=("by_mechanism", "*", "fitness"),
)


@pytest.mark.unit
class TestParallelResolution:
    def test_rows_are_index_major_then_entity(self):
        metrics = _population(
            {
                "m0": {"a": [0.0, 0.1], "b": [1.0, 1.1]},
                "m1": {"a": [0.2, 0.3], "b": [1.2, 1.3]},
                "m2": {"a": [0.4, 0.5], "b": [1.4, 1.5]},
            },
            generations=2,
        )
        table = RecordingReporter()._resolve_parallel(metrics, PARALLEL)
        assert isinstance(table, Table)
        assert table.columns == ("a", "b", "fitness")
        assert table.color_label == "fitness"
        assert len(table.rows) == 6 and len(table.color) == 6
        # generation 0 for m0, m1, m2 then generation 1
        assert table.rows[0] == [0.0, 1.0, 0.0] and table.rows[1] == [0.2, 1.2, 1.0]
        assert table.rows[3] == [0.1, 1.1, 1.0] and table.rows[5] == [0.5, 1.5, 3.0]
        assert table.color == [row[2] for row in table.rows]

    def test_explicit_parameter_paths_and_root_color(self):
        metrics = _population(
            {
                "m0": {"a": [0.0, 0.1], "b": [1.0, 1.1]},
                "m1": {"a": [0.2, 0.3], "b": [1.2, 1.3]},
            },
            generations=2,
        )
        q = ParallelCoordinatesQuery(
            title="pc",
            dimensions=(
                ("by_mechanism", "*", "by_parameter", "b", "value"),
                ("by_mechanism", "*", "by_parameter", "a", "value"),
                ("by_mechanism", "*", "fitness"),
            ),
            color=("generation",),
        )
        table = RecordingReporter()._resolve_parallel(metrics, q)
        assert table.columns == ("b", "a", "fitness")
        assert table.color == [0, 0, 1, 1] and table.color_label == "generation"
        assert table.rows[2] == [1.1, 0.1, 1.0]

    def test_duplicate_axis_label_is_an_error(self):
        metrics = _population({"m0": {"a": [0.0]}}, generations=1)
        q = ParallelCoordinatesQuery(
            title="pc",
            dimensions=(
                ("by_mechanism", "*", "fitness"),
                ("by_mechanism", "*", "fitness"),
            ),
            color=("by_mechanism", "*", "fitness"),
        )
        with pytest.raises(ValueError, match="'fitness'"):
            RecordingReporter()._resolve_parallel(metrics, q)

    def test_misaligned_lengths_are_an_error(self):
        metrics = _population({"m0": {"a": [0.0, 0.1]}}, generations=2)
        metrics.by_mechanism["m0"].fitness.append(5.0)
        with pytest.raises(ValueError, match="not aligned"):
            RecordingReporter()._resolve_parallel(metrics, PARALLEL)

    def test_missing_column_for_an_entity_is_an_error(self):
        metrics = _population({"m0": {"a": [0.0]}, "m1": {"a": [0.1]}}, generations=1)
        metrics.by_mechanism["m1"].by_parameter.pop("a")
        with pytest.raises(ValueError, match="'m1'.*'a'"):
            RecordingReporter()._resolve_parallel(metrics, PARALLEL)

    def test_empty_dynamic_node_yields_an_empty_table(self):
        metrics = MetricLogger.from_schema(_RootS).peek()
        table = RecordingReporter()._resolve_parallel(metrics, PARALLEL)
        assert table.rows == [] and table.color == [] and table.columns == ("fitness",)

    def test_report_dispatches_by_query_type(self, caplog):
        metrics = _population({"m0": {"a": [0.0]}}, generations=1)
        r = RecordingReporter()
        line = Query(
            title="line", x=("generation",), y=("by_mechanism", "*", "fitness")
        )
        r.add_query(line, PARALLEL)
        assert r.queries == (line, PARALLEL)
        with caplog.at_level(logging.WARNING, logger="core.reporting.base"):
            r.report(metrics)
        assert [q for q, _ in r.reports] == [line]
        assert "RecordingReporter does not render parallel coordinates" in caplog.text
        assert "'pc'" in caplog.text


# --- end to end on the real ES schema ---------------------------------------------


def _two_generations() -> ESSchema:
    opt = make_es()
    logger = MetricLogger.from_schema(ESSchema)
    population = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
    )
    for gen, fitness in enumerate(
        [np.array([1.0, 4.0, 2.0, 3.0]), np.array([2.0, 1.0, 5.0, 0.0])]
    ):
        opt.generation = gen
        opt.best_fitness = float(max(opt.best_fitness, fitness.max()))
        opt.best_candidate = population[int(np.argmax(fitness))]
        logger.push_data(
            opt._to_logger_payload(
                inner=InnerSchema(value=float(gen)),
                population=population,
                fitness=fitness,
                mean=np.array([0.4, 0.5]),
                sigma=0.1,
            )
        )
    return logger.peek()


@pytest.mark.unit
class TestESQueries:
    def test_parameter_fitness_queries_are_colored_by_generation(self):
        metrics = _two_generations()
        (q,) = es_parameter_fitness_queries(("fixed_quota",))
        assert q.color == ("generation",)
        series = RecordingReporter()._resolve_query(metrics, q)
        assert len(series) == 4
        assert all(s.color == [0, 1] for s in series)
        assert series[1].x == pytest.approx([0.3, 0.3]) and series[1].y == [4.0, 1.0]

    def test_parallel_coordinates_query(self):
        metrics = _two_generations()
        q = es_parallel_coordinates_query(("fixed_quota", "restoration_subsidy"))
        assert q.title == "Parallel coordinates of evaluated mechanisms"
        table = RecordingReporter()._resolve_parallel(metrics, q)
        assert table.columns == ("fixed_quota", "restoration_subsidy", "fitness")
        assert table.color_label == "fitness"
        assert len(table.rows) == 8  # 4 candidates x 2 generations
        assert table.rows[1] == pytest.approx([0.3, 0.4, 4.0])
        assert table.rows[6] == pytest.approx([0.5, 0.6, 5.0])
        assert table.color == [1.0, 4.0, 2.0, 3.0, 2.0, 1.0, 5.0, 0.0]

    def test_parallel_columns_are_known_before_the_first_generation(self):
        metrics = MetricLogger.from_schema(ESSchema).peek()
        q = es_parallel_coordinates_query(("fixed_quota", "restoration_subsidy"))
        table = RecordingReporter()._resolve_parallel(metrics, q)
        assert table.columns == ("fixed_quota", "restoration_subsidy", "fitness")
        assert table.rows == [] and table.color_label == "fitness"

    def test_parallel_with_explicit_candidate_and_shared_color(self):
        metrics = _two_generations()
        q = ParallelCoordinatesQuery(
            title="one candidate",
            dimensions=(
                ("by_mechanism", "1", "by_parameter", "fixed_quota", "value"),
                ("by_mechanism", "1", "fitness"),
            ),
            color=("generation",),
        )
        table = RecordingReporter()._resolve_parallel(metrics, q)
        # No wildcard: the leaf name labels the axis and there is one anonymous entity.
        assert table.columns == ("value", "fitness")
        assert [pytest.approx(row) for row in table.rows] == [[0.3, 4.0], [0.3, 1.0]]
        assert table.color == [0, 1] and table.color_label == "generation"

    def test_unknown_candidate_id_is_a_key_error(self):
        metrics = _two_generations()
        q = ParallelCoordinatesQuery(
            title="bad",
            dimensions=(("by_mechanism", "zz", "fitness"),),
            color=("generation",),
        )
        with pytest.raises(KeyError, match="no id 'zz'"):
            RecordingReporter()._resolve_parallel(metrics, q)

    def test_unknown_field_on_an_empty_node_is_a_key_error(self):
        metrics = MetricLogger.from_schema(ESSchema).peek()
        q = ParallelCoordinatesQuery(
            title="bad",
            dimensions=(("by_mechanism", "*", "nope", "*", "value"),),
            color=("generation",),
        )
        # The static walk stops at the unknown field; the empty node expands to
        # nothing and the shared colour alone does not make a row.
        table = RecordingReporter()._resolve_parallel(metrics, q)
        assert table.columns == ("value",) and table.rows == [] and table.color == []
        assert Reporter._unwrap_optional(dict[str, int]) == dict[str, int]
        assert Reporter._unwrap_optional(Optional[int]) is int

    def test_color_expanding_to_several_axes_is_an_error(self):
        metrics = _two_generations()
        q = ParallelCoordinatesQuery(
            title="bad",
            dimensions=(("by_mechanism", "*", "fitness"),),
            color=("by_mechanism", "*", "by_parameter", "*", "value"),
        )
        with pytest.raises(ValueError, match="several axes"):
            RecordingReporter()._resolve_parallel(metrics, q)


# --- backends ---------------------------------------------------------------------


X = [0.1, 0.2, 0.3]
COLORED = Query(
    title="Fitness vs q",
    x=("by_mechanism", "*", "by_parameter", "q", "value"),
    y=("by_mechanism", "*", "fitness"),
    color=("generation",),
)
COLORED_SERIES = [
    Series("by_mechanism/0/fitness", X, [1.0, 2.0, 3.0], color=[0, 1, 2]),
    Series("by_mechanism/1/fitness", X, [3.0, 4.0, 5.0], color=[0, 1, 2]),
]
TABLE = Table(
    columns=("a", "b", "fitness"),
    rows=[[0.0, 1.0, 2.0], [1.0, 1.0, 4.0]],
    color=[2.0, 4.0],
    color_label="fitness",
)
EMPTY_TABLE = Table(columns=("a",), rows=[], color=[], color_label="fitness")


@pytest.fixture
def wandb_reporter(monkeypatch):
    logged = []
    run = SimpleNamespace(
        log=lambda payload: logged.append(payload),
        finish=lambda: logged.append("finished"),
    )
    monkeypatch.setattr("core.reporting.wandb.wandb.init", lambda **kw: run)
    monkeypatch.setattr("core.reporting.wandb.wandb.Settings", lambda **kw: kw)
    cfg = WandbConfig(project="p")
    cfg.world = "w"
    cfg.outer_iters = 2
    reporter: WandbReporter = cfg.build(label="x")
    return reporter, logged


@pytest.mark.unit
class TestWandbColor:
    def test_colored_scatter_traces_and_layout(self, wandb_reporter):
        reporter, logged = wandb_reporter
        reporter._report(COLORED, COLORED_SERIES)
        (payload,) = logged
        fig = payload[f"plots/{sanitize_key(COLORED.title)}"]
        assert len(fig.data) == 2
        trace = fig.data[1]
        assert trace.mode == "markers" and trace.name == "by_mechanism/1/fitness"
        assert trace.marker.coloraxis == "coloraxis" and trace.marker.size == 7
        assert list(trace.marker.color) == [0, 1, 2]
        assert list(trace.customdata) == ["by_mechanism/1/fitness"] * 3
        assert "by_mechanism/*/by_parameter/q/value=%{x}" in trace.hovertemplate
        assert "generation=%{marker.color}" in trace.hovertemplate
        assert (
            "value=%{y}" in trace.hovertemplate
            and "%{customdata}" in trace.hovertemplate
        )
        assert fig.layout.coloraxis.colorbar.title.text == "generation"
        assert fig.layout.coloraxis.colorscale[0][1].lower() == "#440154"  # Viridis
        assert fig.layout.hovermode == "closest"

    def test_uncolored_series_keep_unified_hover(self, wandb_reporter):
        reporter, logged = wandb_reporter
        plain = Query(title="plain", x=("iter",), y=("a",))
        reporter._report(plain, [Series("a", [1, 2], [1.0, 2.0])])
        fig = logged[0]["plots/plain"]
        assert (
            fig.layout.hovermode == "x unified" and fig.data[0].mode == "lines+markers"
        )
        assert fig.layout.coloraxis.colorbar.title.text is None


@pytest.mark.unit
class TestWandbParallel:
    def test_parcoords_figure(self, wandb_reporter):
        reporter, logged = wandb_reporter
        reporter._report_table(PARALLEL, TABLE)
        (payload,) = logged
        fig = payload[f"plots/{sanitize_key(PARALLEL.title)}"]
        (trace,) = fig.data
        assert isinstance(trace, go.Parcoords)
        assert [d.label for d in trace.dimensions] == ["a", "b", "fitness"]
        assert list(trace.dimensions[0].values) == [0.0, 1.0]
        assert list(trace.dimensions[0].range) == pytest.approx([-0.05, 1.05])
        assert list(trace.dimensions[1].range) == pytest.approx([0.5, 1.5])  # constant
        assert list(trace.dimensions[2].range) == pytest.approx([1.9, 4.1])
        assert list(trace.line.color) == [2.0, 4.0]
        assert trace.line.colorbar.title.text == "fitness" and trace.line.showscale
        assert fig.layout.title.text == "pc" and fig.layout.height == 650

    def test_empty_table_logs_nothing(self, wandb_reporter):
        reporter, logged = wandb_reporter
        reporter._report_table(PARALLEL, EMPTY_TABLE)
        assert logged == []

    def test_run_failed_to_initialize(self, monkeypatch):
        monkeypatch.setattr("core.reporting.wandb.wandb.init", lambda **kw: None)
        monkeypatch.setattr("core.reporting.wandb.wandb.Settings", lambda **kw: kw)
        reporter = WandbReporter(project="p", name="n", run_id="id", group="g")
        with pytest.raises(RuntimeError, match="failed to initialize"):
            reporter._report_table(PARALLEL, TABLE)


@pytest.mark.unit
class TestCSVColorAndTable:
    def test_color_column(self, tmp_path):
        reporter = CSVReporter(output_dir=tmp_path)
        reporter._report(COLORED, COLORED_SERIES)
        with reporter.path_for(COLORED).open() as f:
            rows = list(csv.DictReader(f))
        assert "color" in rows[0]
        assert [
            int(r["color"]) for r in rows if r["series"].endswith("/1/fitness")
        ] == [
            0,
            1,
            2,
        ]
        reporter._report(COLORED, [Series("a", [1], [1.0])])
        with reporter.path_for(COLORED).open() as f:
            (row,) = list(csv.DictReader(f))
        assert row["color"] == ""

    def test_wide_table_file(self, tmp_path):
        reporter = CSVReporter(output_dir=tmp_path)
        reporter._report_table(PARALLEL, TABLE)
        path = tmp_path / f"{sanitize_key(PARALLEL.title)}.csv"
        with path.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["a", "b", "fitness", "color:fitness"]
        assert rows[1:] == [["0.0", "1.0", "2.0", "2.0"], ["1.0", "1.0", "4.0", "4.0"]]
        reporter._report_table(PARALLEL, TABLE)  # rewritten, not appended
        with path.open() as f:
            assert sum(1 for _ in f) == 3
        reporter._report_table(PARALLEL, EMPTY_TABLE)
        with path.open() as f:
            assert sum(1 for _ in f) == 3  # untouched


@pytest.mark.unit
class TestTensorBoardColorAndTable:
    def test_parallel_query_is_skipped_with_a_warning(self, tmp_path, caplog):
        reporter = TensorBoardReporter(log_dir=tmp_path)
        calls = []
        reporter._writer = SimpleNamespace(
            add_scalar=lambda **kw: calls.append(kw),
            flush=lambda: None,
            close=lambda: None,
        )
        with caplog.at_level(logging.WARNING, logger="core.reporting.base"):
            reporter._report_table(PARALLEL, TABLE)
        assert calls == []
        assert "TensorBoardReporter does not render parallel coordinates" in caplog.text

    def test_color_is_ignored_with_an_info(self, tmp_path, caplog):
        reporter = TensorBoardReporter(log_dir=tmp_path)
        calls = []
        reporter._writer = SimpleNamespace(
            add_scalar=lambda **kw: calls.append(kw),
            flush=lambda: None,
            close=lambda: None,
        )
        with caplog.at_level(logging.INFO, logger="core.reporting.tensor_board"):
            reporter._report(
                COLORED,
                [Series("by_mechanism/0/fitness", [1, 2], [1.0, 2.0], color=[0, 1])],
            )
        assert len(calls) == 2
        assert "ignores the color" in caplog.text and "Fitness vs q" in caplog.text
