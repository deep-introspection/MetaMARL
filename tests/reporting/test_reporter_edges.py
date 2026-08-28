"""Edge cases of the reporting stack: query validation, grouping guards, backends.

Complements ``test_query_and_reporter.py`` and ``test_backends.py`` with the
error paths: empty query paths, unknown reductions, misaligned replicates,
schema getter, TensorBoard writer lifecycle on a real ``tmp_path`` and the
W&B reporter against a fully mocked ``wandb`` module (no network, no run
directory).
"""

from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.reporting.base import Reporter
from core.reporting.enums import ReporterType, Resolution
from core.reporting.query import Query, Series
from core.reporting.tensor_board import TensorBoardConfig, TensorBoardReporter
from core.reporting.wandb import WandbConfig, WandbReporter
from tests.metrics.conftest import RootSchema


class RecordingReporter(Reporter):
    def __init__(self):
        self.reports = []

    def _report(self, query, series):
        self.reports.append((query, series))

    def close(self):
        pass


@pytest.mark.unit
class TestQueryValidation:
    def test_empty_x_is_rejected(self):
        with pytest.raises(ValueError, match="x path must not be empty"):
            Query(title="t", x=(), y=("a",))

    def test_empty_y_is_rejected(self):
        with pytest.raises(ValueError, match="y paths must not be empty"):
            Query(title="t", x=("iter",), y=())
        with pytest.raises(ValueError, match="y paths must not be empty"):
            Query(title="t", x=("iter",), y=(("a",), ()))

    def test_has_wildcards(self):
        assert not Query(title="t", x=("iter",), y=("a", "b")).has_wildcards
        assert Query(title="t", x=("iter",), y=("a", "*", "b")).has_wildcards
        assert Query(title="t", x=("a", "*"), y=("b",)).has_wildcards
        assert Query(title="t", x=("iter",), y=(("a",), ("b", "*"))).has_wildcards

    def test_series_defaults(self):
        s = Series("lbl", [1], [2.0])
        assert s.error is None
        with pytest.raises(AttributeError):
            s.label = "other"


@pytest.mark.unit
class TestEnums:
    def test_values(self):
        assert ReporterType("wandb") is ReporterType.wandb
        assert ReporterType.local.value == "local"
        assert {r.value for r in Resolution} == {
            "env_steps",
            "train_iters",
            "generation",
        }


class _SeedS(MetricSchema):
    value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    step: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )


class _MechS(MetricSchema):
    by_seed: dict[str, _SeedS] = Field(default_factory=dict)


class _RootS(MetricSchema):
    by_mechanism: dict[str, _MechS] = Field(default_factory=dict)


def _nested_with_per_seed_x(seed_lengths: dict[str, int]):
    """One mechanism, seeds with their own ``step`` axis of the given lengths."""
    logger = MetricLogger.from_schema(_RootS)
    for seed, length in seed_lengths.items():
        for it in range(length):
            logger.push(("by_mechanism", "m0", "by_seed", seed, "step"), it)
            logger.push(("by_mechanism", "m0", "by_seed", seed, "value"), float(it))
    return logger.peek()


@pytest.mark.unit
class TestReporterEdges:
    def test_schema_getter(self):
        r = RecordingReporter()
        assert r.schema is None
        r.schema = RootSchema
        assert r.schema is RootSchema

    def test_unknown_reduction_is_rejected_at_resolution(self):
        # ``Query`` does not validate the Literal at construction; the reporter does.
        q = Query(title="t", x=("iter",), y=("static", "series_value"), reduce="max")
        logger = MetricLogger.from_schema(RootSchema)
        logger.push(("iter",), 0)
        logger.push(("static", "series_value"), 1.0)
        with pytest.raises(ValueError, match="Unknown query reduction"):
            RecordingReporter()._resolve_query(logger.peek(), q)

    def test_replicates_with_different_x_are_not_aligned(self):
        metrics = _nested_with_per_seed_x({"s1": 3, "s2": 2})
        q = Query(
            title="t",
            x=("by_mechanism", "*", "by_seed", "*", "step"),
            y=("by_mechanism", "*", "by_seed", "*", "value"),
            reduce="mean",
        )
        with pytest.raises(ValueError, match="not aligned"):
            RecordingReporter()._resolve_query(metrics, q)

    def test_replicates_with_matching_per_seed_x_are_averaged(self):
        metrics = _nested_with_per_seed_x({"s1": 3, "s2": 3})
        q = Query(
            title="t",
            x=("by_mechanism", "*", "by_seed", "*", "step"),
            y=("by_mechanism", "*", "by_seed", "*", "value"),
            reduce="mean",
            error="std",
        )
        (s,) = RecordingReporter()._resolve_query(metrics, q)
        assert s.label == "m0" and s.x == [0, 1, 2] and s.y == [0.0, 1.0, 2.0]
        assert s.error == [0.0, 0.0, 0.0]

    def test_mean_without_error_has_no_band(self):
        metrics = _nested_with_per_seed_x({"s1": 2, "s2": 2})
        q = Query(
            title="t",
            x=("by_mechanism", "m0", "by_seed", "s1", "step"),
            y=("by_mechanism", "*", "by_seed", "*", "value"),
            reduce="mean",
        )
        (s,) = RecordingReporter()._resolve_query(metrics, q)
        assert s.error is None and s.y == [0.0, 1.0]

    def test_wildcard_over_empty_dynamic_node_yields_no_series(self):
        metrics = MetricLogger.from_schema(_RootS).peek()
        q = Query(
            title="t",
            x=("iter",),
            y=("by_mechanism", "*", "by_seed", "*", "value"),
            reduce="mean",
            error="std",
        )
        r = RecordingReporter()
        assert r._resolve_query(metrics, q) == []
        r.add_query(q)
        r.report(metrics)
        assert r.reports == [(q, [])]  # backends decide to skip empty series


RAW = Query(title="Raw", x=("iter",), y=("a",))
X = [1, 2]


@pytest.mark.unit
class TestTensorBoardLifecycle:
    def test_log_dir_property_and_close_without_writer(self, tmp_path):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        assert reporter.log_dir == tmp_path / "tb"
        reporter.close()  # no writer yet: nothing to release
        assert reporter._writer is None

    def test_empty_series_does_not_create_a_writer(self, tmp_path):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        reporter._report(RAW, [])
        assert reporter._writer is None
        assert not (tmp_path / "tb").exists()

    def test_config_without_label(self, tmp_path):
        cfg = TensorBoardConfig(project="p", log_dir=tmp_path)
        cfg.world = "w"
        assert cfg.build().log_dir == tmp_path / "p" / "w"
        assert cfg.build(label="env").log_dir == tmp_path / "p" / "w-env"

    def test_writer_is_created_once_and_reset_on_close(self, tmp_path):
        pytest.importorskip("tensorboard")
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        reporter._report(RAW, [Series("a", X, [1.0, 2.0])])
        writer = reporter._writer
        assert writer is not None
        reporter._report(RAW, [Series("a", X, [3.0, 4.0])])
        assert reporter._writer is writer
        reporter.close()
        assert reporter._writer is None
        assert list((tmp_path / "tb").glob("events.out.tfevents.*"))

    def test_std_band_and_step_from_float_integer(self, tmp_path):
        reporter = TensorBoardReporter(log_dir=tmp_path)
        calls = []
        reporter._writer = SimpleNamespace(
            add_scalar=lambda **kw: calls.append(kw),
            flush=lambda: calls.append("flush"),
            close=lambda: calls.append("close"),
        )
        reporter._report(RAW, [Series("a", [1.0, 2.0], [1.0, 2.0], error=[0.5, 0.5])])
        assert [c["global_step"] for c in calls if c != "flush"] == [1, 2, 1, 2]
        assert calls[-1] == "flush"
        reporter.close()
        assert calls[-1] == "close" and reporter._writer is None


@pytest.mark.unit
class TestWandbMocked:
    def test_run_failed_to_initialize(self, monkeypatch):
        monkeypatch.setattr("core.reporting.wandb.wandb.init", lambda **kw: None)
        monkeypatch.setattr("core.reporting.wandb.wandb.Settings", lambda **kw: kw)
        reporter = WandbReporter(project="p", name="n", run_id="id", group="g")
        with pytest.raises(RuntimeError, match="failed to initialize"):
            reporter._report(RAW, [Series("a", X, [1.0, 2.0])])

    def test_init_arguments_and_lazy_run(self, monkeypatch):
        seen = {}

        def fake_init(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(log=lambda p: None, finish=lambda: None)

        monkeypatch.setattr("core.reporting.wandb.wandb.init", fake_init)
        monkeypatch.setattr("core.reporting.wandb.wandb.Settings", lambda **kw: kw)
        cfg = WandbConfig(project="p")
        cfg.world = "w"
        cfg.outer_iters = 3
        reporter = cfg.build()  # no label -> run name is the world name
        assert reporter._name == "w" and seen == {}  # nothing touched yet
        reporter._report(RAW, [Series("a", X, [1.0, 2.0])])
        assert seen["project"] == "p" and seen["group"] == "w"
        assert seen["config"] == {"outer_iters": 3, "world_name": "w"}
        assert seen["settings"]["quiet"] is True
        assert seen["reinit"] == "create_new"
        reporter._report(RAW, [Series("a", X, [1.0, 2.0])])
        assert seen["id"] == reporter._run_id  # a single init for both reports

    def test_figure_without_a_run(self):
        fig = WandbReporter._figure(
            RAW,
            [
                Series("a ±1 std", X, [1.0, 2.0], error=[0.1, 0.1]),
                Series("b", X, [3.0, 4.0]),
            ],
        )
        assert [t.name for t in fig.data] == ["a ±1 std", "a ±1 std", "b"]
        assert fig.data[2].line.width == 2 and fig.data[1].line.width == 3
        assert WandbReporter._path_name(("a", "b")) == "a/b"
