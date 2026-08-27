"""``BilevelOptimizer.run`` orchestration with stub inner/outer optimizers (no Ray)."""

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


def make_config(outer_iters):
    return (
        BilevelConfig()
        .world(world_name="w")
        .mechanism(mechanism=SubsidyMechanism(subsidy=0.1, cost=0.1))
        .training(outer_iters=outer_iters)
    )


@pytest.mark.unit
def test_run_iterates_outer_and_reports():
    outer = StubOuter(
        [{"best_fitness": 1.0}, {"best_fitness": 2.0}, {"best_fitness": 1.5}]
    )
    opt = BilevelOptimizer(make_config(3), outer=outer, inner=object())
    result = opt.run()
    assert outer.calls == 3
    assert result["converged"] is False
    assert result["outer_iters"] == 3
    assert result["best_fitness"] == 2.0
    np.testing.assert_allclose(result["best_mechanism"], [0.5])
    assert opt.mechanism_template.dimension == 1


@pytest.mark.unit
def test_run_stops_early_on_convergence():
    outer = StubOuter(
        [
            {"best_fitness": 1.0},
            {"best_fitness": 2.0, "converged": True},
            {"best_fitness": 9.0},
        ]
    )
    opt = BilevelOptimizer(make_config(3), outer=outer, inner=object())
    result = opt.run()
    assert outer.calls == 2
    assert result["converged"] is True and result["outer_iters"] == 2
