"""``BilevelOptimizer.run`` orchestration with stub inner/outer optimizers (no Ray).

The outer loop calls ``outer.run()`` up to ``outer_iters`` times, stops early
when the outer optimizer reports convergence, closes the primary reporter and
returns a summary dict (``run`` is annotated ``-> None`` in production but
does return that dict).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.optimizers.bilevel import BilevelConfig, BilevelOptimizer


class StubOuter:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.best_fitness = -np.inf
        self.best_candidate = np.array([0.5])

    def run(self):
        result = self._results[self.calls]
        self.calls += 1
        self.best_fitness = max(self.best_fitness, result["best_fitness"])
        return result


class StubReporter:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def make_config(outer_iters):
    return (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1))
        .training(outer_iters=outer_iters, output_dir="out")
    )


@pytest.mark.unit
def test_constructor_copies_config_into_attributes():
    cfg = make_config(4)
    inner = object()
    opt = BilevelOptimizer(cfg, outer=StubOuter([]), inner=inner, reporter=None)
    assert opt.config is cfg and opt.inner is inner
    assert opt.world_name == cfg.world_name
    assert opt.max_outer_iters == 4 and opt.output_dir == "out"
    assert opt.mechanism_template is cfg.mechanism_template
    assert opt.outer_iter == 0 and opt.converged is False
    assert opt.all_trajectories == [] and opt.population_history == []
    assert opt.es_metrics_history == [] and opt.reporting is None
    assert opt.env is None and opt.world is None


@pytest.mark.unit
def test_run_iterates_outer_closes_reporter_and_reports(caplog):
    outer = StubOuter(
        [{"best_fitness": 1.0}, {"best_fitness": 2.0}, {"best_fitness": 1.5}]
    )
    reporter = StubReporter()
    opt = BilevelOptimizer(
        make_config(3), outer=outer, inner=object(), reporter=reporter
    )
    with caplog.at_level(logging.INFO, logger="core.optimizers.bilevel"):
        result = opt.run()
    assert outer.calls == 3
    assert reporter.closed == 1
    assert result == {
        "converged": False,
        "outer_iters": 3,
        "best_fitness": 2.0,
        "best_mechanism": outer.best_candidate,
        "all_trajectories": [],
        "population_history": [],
    }
    assert opt.outer_iter == 2
    messages = [r.getMessage() for r in caplog.records]
    assert any("Starting run" in m for m in messages)
    assert any("Outer iteration 3 / 3" in m for m in messages)
    assert any("Run finished" in m and "converged=False" in m for m in messages)


@pytest.mark.unit
def test_run_stops_early_on_convergence_without_reporter(caplog):
    outer = StubOuter(
        [
            {"best_fitness": 1.0},
            {"best_fitness": 2.0, "converged": True},
            {"best_fitness": 9.0},
        ]
    )
    opt = BilevelOptimizer(make_config(3), outer=outer, inner=object(), reporter=None)
    with caplog.at_level(logging.INFO, logger="core.optimizers.bilevel"):
        result = opt.run()
    assert outer.calls == 2
    assert result["converged"] is True and result["outer_iters"] == 2
    assert result["best_fitness"] == 2.0
    assert any("EARLY STOP" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_run_with_zero_iterations_reports_one_iteration():
    """``outer_iters=0`` never runs the outer loop but the summary says 1.

    The summary uses ``outer_iter + 1`` with ``outer_iter`` initialised to 0,
    so an empty run is reported as one iteration; documented, not fixed.
    """
    outer = StubOuter([])
    result = BilevelOptimizer(
        make_config(0), outer=outer, inner=object(), reporter=None
    ).run()
    assert outer.calls == 0
    assert result["outer_iters"] == 1
    assert result["best_fitness"] == -np.inf
