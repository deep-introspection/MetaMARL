"""Envs and configs wiring the metric logger and reporter (logging branch)."""

from typing import Optional

import pytest
from pydantic import Field

from core.envs.base import BaseEnv
from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema
from core.optimizers.es.config import ESConfig
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query


class StepSchema(MetricSchema):
    value: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )


class RecordingReporter(Reporter):
    def __init__(self, label):
        self.label = label
        self.reports = []

    def _report(self, query, x, ys):
        self.reports.append((query.title, x, ys))

    def close(self):
        pass


class RecordingConfig(ReporterConfig):
    built = []

    def build(self, *, label=None):
        reporter = RecordingReporter(label)
        RecordingConfig.built.append(reporter)
        return reporter


class CountingEnv(BaseEnv):
    def _pre_reset(self, seed=None):
        pass

    def _reset(self):
        return 0

    def _step(self, action):
        self._log(("value",), float(action))
        return action, 0.0, False, False, {}


@pytest.mark.unit
def test_env_without_schema_or_reporter_is_silent(fake_world):
    env = CountingEnv(world=fake_world)
    assert env.logger is None and env.reporter is None
    env.reset()
    env.step(1)  # _log is a no-op


@pytest.mark.unit
def test_env_logs_iter_and_reports_queries(fake_world):
    RecordingConfig.built.clear()
    cfg = RecordingConfig(project="p")
    cfg.world = "w"
    env = CountingEnv(
        world=fake_world,
        env_name="counting",
        mode="train",
        policy_seed=3,
        seed=5,
        reporter_cfg=cfg,
        schema=StepSchema,
        queries=(Query(title="v", x=("iter",), y=("value",)),),
    )
    assert env.reporter.label == "counting|mode=train|ps=3|ss=5"
    assert env.reporter.schema is StepSchema and len(env.reporter.queries) == 1

    env.reset()
    for a in (1.0, 2.0):
        env.step(a)
    peeked = env.logger.peek()
    assert peeked.iter == [0, 1, 2]  # reset + 2 steps
    assert peeked.value == [1.0, 2.0]
    # iter has one more entry than value: the query needs aligned lengths
    with pytest.raises(ValueError, match="equal length"):
        env.reporter.report(peeked)
    env.logger.flush(("iter",))
    env.logger.push(("iter",), 1)
    env.logger.push(("iter",), 2)
    env.reporter.report(env.logger.peek())
    assert env.reporter.reports == [("v", [1, 2], [[1.0, 2.0]])]


@pytest.mark.unit
def test_optimizer_config_reporting_plumbing(fake_world):
    RecordingConfig.built.clear()

    class AnalyticEnv(CountingEnv):
        pass

    cfg = (
        ESConfig()
        .training(sigma=0.1)
        .environment(
            env=AnalyticEnv,
            schema=StepSchema,
            queries=[Query(title="env-q", x=("iter",), y=("value",))],
            env_config={"mechanism_space": None},
        )
        .reporting(
            schema=StepSchema,
            queries=(Query(title="opt-q", x=("iter",), y=("value",)),),
        )
    )
    cfg.dimension = 1
    cfg.reporter_cfg = RecordingConfig(project="p")

    assert cfg._reporting_queries_env[0].title == "env-q"
    assert cfg._reporting_queries[0].title == "opt-q"

    # ESOptimizer._on_env_init needs a mechanism space; give the env one lazily
    opt_env_holder = {}

    class SpaceEnv(AnalyticEnv):
        def __init__(self, **kw):
            super().__init__(**kw)
            from types import SimpleNamespace

            self.m_space = SimpleNamespace(optimize_params=["p"])
            opt_env_holder["env"] = self

    cfg.env = SpaceEnv
    opt = cfg.build_optimizer(world=fake_world)

    # optimizer-level reporter got the optimizer queries, env got the env queries
    assert opt.reporting.label == "ESOptimizer"
    assert [q.title for q in opt.reporting.queries] == ["opt-q"]
    env = opt_env_holder["env"]
    assert [q.title for q in env.reporter.queries] == ["env-q"]
    assert env.logger is not None and env.logger._schema is StepSchema
    assert env.reporter is not opt.reporting
    assert opt.reporting is not None and opt.logger is not None

    # report_metrics is a no-op without reporter, and reduce/flush work
    opt.reporting = None
    opt.report_metrics()
    opt.flush_metrics()
    assert opt.reduce_metrics().__class__.__name__ == "ESSchema"


@pytest.mark.unit
def test_optimizer_config_without_reporter(fake_world):
    class SpaceEnv(CountingEnv):
        def __init__(self, **kw):
            super().__init__(**kw)
            from types import SimpleNamespace

            self.m_space = SimpleNamespace(optimize_params=["p"])

    cfg = ESConfig().training(sigma=0.1).environment(env=SpaceEnv)
    cfg.dimension = 1
    opt = cfg.build_optimizer(world=fake_world)
    assert opt.reporting is None and opt.env.reporter is None and opt.env.logger is None
