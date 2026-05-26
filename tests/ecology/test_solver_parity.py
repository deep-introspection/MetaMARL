"""Cross-check Euler and RK45 agree for small dt."""

import pytest

from bilevel_fishery.ecology.dynamics import step
from bilevel_fishery.ecology.params import EcologyParams
from bilevel_fishery.ecology.state import EcologicalState


@pytest.mark.unit
def test_euler_and_rk45_agree_for_small_dt() -> None:
    """At ``dt = 0.001`` over 1 time unit, both integrators agree within 1%."""
    p_euler = EcologyParams(integrator="euler", dt=0.001)
    p_rk45 = EcologyParams(integrator="rk45", dt=0.001)

    state_euler = EcologicalState(fish=10.0, algae=20.0)
    state_rk45 = EcologicalState(fish=10.0, algae=20.0)

    for _ in range(1000):
        state_euler = step(state_euler, p_euler, harvest=0.0)
        state_rk45 = step(state_rk45, p_rk45, harvest=0.0)

    assert state_euler.fish == pytest.approx(state_rk45.fish, rel=0.01)
    assert state_euler.algae == pytest.approx(state_rk45.algae, rel=0.01)


@pytest.mark.unit
def test_euler_diverges_from_rk45_for_large_dt() -> None:
    """At ``dt = 0.2`` off-equilibrium, Euler accumulates substantial error vs RK45.

    Starting from the equilibrium ``(F*, A*) = (10, 20)`` would leave both
    integrators stationary — the divergence only becomes visible when the
    trajectory has actual curvature, which requires being off-equilibrium.
    """
    p_euler = EcologyParams(integrator="euler", dt=0.2)
    p_rk45 = EcologyParams(integrator="rk45", dt=0.2)

    # Off-equilibrium initial state (equilibrium is at (10, 20)).
    state_euler = EcologicalState(fish=15.0, algae=15.0)
    state_rk45 = EcologicalState(fish=15.0, algae=15.0)

    # ~10 time units of evolution
    for _ in range(50):
        state_euler = step(state_euler, p_euler, harvest=0.0)
        state_rk45 = step(state_rk45, p_rk45, harvest=0.0)

    # Relative difference should be >5% — Euler accumulates oscillation amplitude
    rel_diff_fish = abs(state_euler.fish - state_rk45.fish) / max(state_rk45.fish, 1e-8)
    assert rel_diff_fish > 0.05
