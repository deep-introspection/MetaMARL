"""Unit tests for ``log_and_report_episode_metrics`` (the episode-end hook).

The hook is checked with a real ``MetricLogger`` and the recording reporter of
the env logging tests: peek, report, reduce, then hand the reduced schema to
RLlib's ``MetricsLogger``. The episode-tagging callback and the single-round
evaluation function are covered in ``tests/unit/test_callbacks.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.callbacks import log_and_report_episode_metrics
from core.metrics.logger import MetricLogger
from core.reporting.query import Query
from tests.envs.test_env_logging import RecordingReporter, StepSchema

# --------------------------------------------------------------------------- #
# log_and_report_episode_metrics
# --------------------------------------------------------------------------- #


def make_logging_env(values=(2.0,), iteration=1):
    """Return ``(env, env_runner)`` with a populated ``MetricLogger``."""
    logger = MetricLogger.from_schema(StepSchema)
    logger.push(("iter",), iteration)
    for value in values:
        logger.push(("value",), value)
    reporter = RecordingReporter("env")
    reporter.add_query(Query(title="v", x=("iter",), y=("value",)))
    env = SimpleNamespace(logger=logger, reporter=reporter)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(envs=[SimpleNamespace(unwrapped=env)])
    )
    return env, env_runner


@pytest.mark.unit
def test_episode_end_callback_reports_then_reduces():
    env, env_runner = make_logging_env(values=(2.0,))
    logged = []
    metrics_logger = SimpleNamespace(log_value=lambda **kw: logged.append(kw))

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="env=0|m=1|ps=2|ss=3|raw=abc"),
        env_runner=env_runner,
        env=None,
        env_index=0,
        metrics_logger=metrics_logger,
    )

    assert env.reporter.reports == [("v", [1], [[2.0]])]
    assert len(env.logger._refs[("value",)]) == 0  # reduced (destructive)
    (call,) = logged
    assert call["key"] == ("by_episode", "env=0|m=1|ps=2|ss=3")
    assert call["reduce"] == "item"
    assert call["value"].value == [2.0] and call["value"].iter == 1


@pytest.mark.unit
def test_episode_end_callback_picks_sub_env_by_index_and_keeps_untagged_id():
    env, _ = make_logging_env(values=(3.0,), iteration=4)
    other = SimpleNamespace(logger=None, reporter=None)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(
            envs=[SimpleNamespace(unwrapped=other), SimpleNamespace(unwrapped=env)]
        )
    )
    logged = []
    metrics_logger = SimpleNamespace(log_value=lambda **kw: logged.append(kw))

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="plain-id"),
        env_runner=env_runner,
        env=None,
        env_index=1,
        metrics_logger=metrics_logger,
        extra="ignored",
    )

    # An ID without the ``|raw=`` marker is used unchanged.
    assert logged[0]["key"] == ("by_episode", "plain-id")
    assert logged[0]["value"].value == [3.0]
    assert env.reporter.reports == [("v", [4], [[3.0]])]
