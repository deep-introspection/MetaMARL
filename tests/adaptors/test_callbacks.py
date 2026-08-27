"""``log_and_report_episode_metrics``: peek -> report -> reduce -> hand off to RLlib's logger."""

from types import SimpleNamespace

import pytest

from core.callbacks import log_and_report_episode_metrics
from core.metrics.logger import MetricLogger
from tests.envs.test_env_logging import RecordingReporter, StepSchema


@pytest.mark.unit
def test_episode_end_callback_reports_then_reduces():
    logger = MetricLogger.from_schema(StepSchema)
    logger.push(("iter",), 1)
    logger.push(("value",), 2.0)
    reporter = RecordingReporter("env")
    reporter.add_query(
        __import__("core.reporting.query", fromlist=["Query"]).Query(
            title="v", x=("iter",), y=("value",)
        )
    )
    env = SimpleNamespace(logger=logger, reporter=reporter)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(envs=[SimpleNamespace(unwrapped=env)])
    )
    logged = []
    metrics_logger = SimpleNamespace(log_value=lambda **kw: logged.append(kw))

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="env=0|m=1|ps=2|ss=3|raw=abc"),
        env_runner=env_runner,
        env=None,
        env_index=0,
        metrics_logger=metrics_logger,
    )

    assert reporter.reports == [("v", [1], [[2.0]])]
    assert len(logger._refs[("value",)]) == 0  # reduced (destructive)
    (call,) = logged
    assert call["key"] == ("by_episode", "env=0|m=1|ps=2|ss=3")
    assert call["reduce"] == "item"
    assert call["value"].value == [2.0] and call["value"].iter == 1
