"""CSV, TensorBoard and W&B reporters on the shared ``_report`` contract (TODO §10, §14, §15)."""

import csv
from types import SimpleNamespace

import pytest

from core.reporting.config import ReporterConfig
from core.reporting.csv import CSVConfig, CSVReporter
from core.reporting.query import Query
from core.reporting.tensor_board import TensorBoardConfig, TensorBoardReporter
from core.reporting.wandb import WandbConfig, WandbReporter
from core.utils import sanitize_key

X = [1, 2, 3]
YS = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
RAW = Query(title="Raw series", x=("iter",), y=(("a", "v"), ("b", "v")))
MEAN = Query(
    title="Mean ± std",
    x=("iter",),
    y=(("a", "v"), ("b", "v")),
    reduce="mean",
    error="std",
)


def configure(cfg: ReporterConfig):
    cfg.world = "w"
    cfg.outer_iters = 5
    return cfg


@pytest.mark.unit
class TestReporterConfig:
    def test_copy_and_labels(self, tmp_path):
        cfg = configure(CSVConfig(project="p", output_dir=tmp_path))
        cp = cfg.copy()
        assert cp is not cfg and cp.world == "w" and cp.outer_iters == 5
        assert cfg.build(label="env").output_dir == tmp_path / "p" / "w-env"
        assert cfg.build().output_dir == tmp_path / "p" / "w"


@pytest.mark.unit
class TestCSV:
    def test_raw_and_mean_export_reload(self, tmp_path):
        reporter: CSVReporter = configure(
            CSVConfig(project="p", output_dir=tmp_path)
        ).build(label="x")
        reporter.add_query(RAW, MEAN)
        reporter._report(RAW, X, YS)
        reporter._report(MEAN, X, YS)

        with reporter.path_for(RAW).open() as f:
            rows = list(csv.DictReader(f))
        assert {r["series"] for r in rows} == {"a/v", "b/v"}
        assert [float(r["value"]) for r in rows if r["series"] == "b/v"] == [
            3.0,
            4.0,
            5.0,
        ]
        assert [int(r["x"]) for r in rows if r["series"] == "a/v"] == X

        with reporter.path_for(MEAN).open() as f:
            rows = list(csv.DictReader(f))
        assert [float(r["value"]) for r in rows if r["series"] == "mean"] == [
            2.0,
            3.0,
            4.0,
        ]
        assert [float(r["value"]) for r in rows if r["series"] == "std"] == [
            1.0,
            1.0,
            1.0,
        ]

        # re-reporting overwrites (no duplicate header / rows)
        reporter._report(RAW, X, YS)
        with reporter.path_for(RAW).open() as f:
            assert sum(1 for _ in f) == 1 + 6
        reporter._report(RAW, X, [])  # nothing to write
        reporter.close()

    def test_length_mismatch_is_an_error(self, tmp_path):
        reporter = CSVReporter(output_dir=tmp_path)
        with pytest.raises(ValueError):
            reporter._report(RAW, X, [[1.0, 2.0], [3.0, 4.0, 5.0]])


@pytest.mark.unit
class TestTensorBoard:
    def test_scalars_are_written(self, tmp_path):
        pytest.importorskip("tensorboard")
        reporter: TensorBoardReporter = configure(
            TensorBoardConfig(project="p", log_dir=tmp_path)
        ).build(label="x")
        reporter._report(RAW, X, YS)
        reporter._report(MEAN, X, YS)
        reporter.close()
        assert reporter._writer is None
        files = list((tmp_path / "p" / "w-x").glob("events.out.tfevents.*"))
        assert files and files[0].stat().st_size > 0

    def test_tags_and_integer_steps(self, tmp_path):
        reporter = TensorBoardReporter(log_dir=tmp_path)
        calls = []
        reporter._writer = SimpleNamespace(
            add_scalar=lambda **kw: calls.append(kw),
            flush=lambda: None,
            close=lambda: None,
        )
        reporter._report(RAW, X, YS)
        raw = sanitize_key(RAW.title)
        assert {c["tag"] for c in calls} == {f"{raw}/a/v", f"{raw}/b/v"}
        assert [c["global_step"] for c in calls if c["tag"].endswith("a/v")] == X
        calls.clear()
        reporter._report(MEAN, X, YS)
        mean = sanitize_key(MEAN.title)
        assert {c["tag"] for c in calls} == {f"{mean}/mean", f"{mean}/std"}
        with pytest.raises(TypeError, match="integer-valued"):
            reporter._report(RAW, [0.5, 1.5, 2.5], YS)


@pytest.mark.unit
class TestWandb:
    @pytest.fixture
    def reporter(self, monkeypatch):
        logged = []
        run = SimpleNamespace(
            log=lambda payload: logged.append(payload),
            finish=lambda: logged.append("finished"),
        )
        monkeypatch.setattr("core.reporting.wandb.wandb.init", lambda **kw: run)
        monkeypatch.setattr("core.reporting.wandb.wandb.Settings", lambda **kw: kw)
        reporter: WandbReporter = configure(
            WandbConfig(project="p", quiet=False)
        ).build(label="x")
        assert reporter._name == "w-x" and reporter._settings["quiet"] is False
        return reporter, logged

    def test_raw_query_traces(self, reporter):
        reporter, logged = reporter
        reporter._report(RAW, X, YS)
        (payload,) = logged
        fig = payload[f"plots/{sanitize_key(RAW.title)}"]
        assert [t.name for t in fig.data] == ["a/v", "b/v"]
        assert list(fig.data[1].y) == [3.0, 4.0, 5.0] and list(fig.data[1].x) == X
        assert (
            fig.layout.title.text == "Raw series"
            and fig.layout.xaxis.title.text == "iter"
        )

    def test_mean_query_has_band_and_mean(self, reporter):
        reporter, logged = reporter
        reporter._report(MEAN, X, YS)
        fig = logged[0][f"plots/{sanitize_key(MEAN.title)}"]
        names = [t.name for t in fig.data]
        assert names == ["±1 std", "mean"]
        assert list(fig.data[1].y) == [2.0, 3.0, 4.0]
        band = list(fig.data[0].y)
        assert band[:3] == [3.0, 4.0, 5.0] and band[3:] == [
            3.0,
            2.0,
            1.0,
        ]  # upper then reversed lower

    def test_empty_and_close(self, reporter):
        reporter, logged = reporter
        reporter._report(RAW, X, [])
        assert logged == []
        reporter._init_run()
        reporter.close()
        assert logged == ["finished"] and reporter._run is None
        reporter.close()  # idempotent
