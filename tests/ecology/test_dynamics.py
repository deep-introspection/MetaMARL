"""Numerical property tests for the Lotka-Volterra dynamics."""

import numpy as np
import pytest

from bilevel_fishery.ecology.dynamics import (
    EcologyInstabilityError,
    reset_state,
    step,
)
from bilevel_fishery.ecology.params import EcologyParams
from bilevel_fishery.ecology.state import EcologicalState


@pytest.mark.unit
def test_reset_state_deterministic_without_noise() -> None:
    p = EcologyParams(noise_std=0.0, fish_init=10.0, algae_init=20.0)
    s = reset_state(p, np.random.default_rng(42))
    assert s.fish == 10.0
    assert s.algae == 20.0


@pytest.mark.unit
def test_reset_state_seeded_reproducible() -> None:
    p = EcologyParams(noise_std=0.05)
    s1 = reset_state(p, np.random.default_rng(42))
    s2 = reset_state(p, np.random.default_rng(42))
    assert s1 == s2


@pytest.mark.unit
def test_state_is_immutable() -> None:
    s = EcologicalState(fish=10.0, algae=20.0)
    with pytest.raises(AttributeError):
        s.fish = 5.0  # type: ignore[misc]


@pytest.mark.unit
def test_no_algae_means_fish_decays_exponentially() -> None:
    """With no algae and no harvest, fish should decay as F(0) * exp(-gamma * t)."""
    p = EcologyParams(dt=0.1, integrator="rk45")
    state = EcologicalState(fish=10.0, algae=0.0)
    next_state = step(state, p, harvest=0.0)

    expected_fish = 10.0 * np.exp(-p.gamma * p.dt)
    assert next_state.fish == pytest.approx(expected_fish, rel=1e-4)
    assert next_state.algae == 0.0


@pytest.mark.unit
def test_no_fish_means_algae_grows_exponentially() -> None:
    """With ``fish = 0``, algae should follow ``A(t) = A(0) exp(alpha t)``."""
    p = EcologyParams(dt=0.1, integrator="rk45")
    state = EcologicalState(fish=0.0, algae=10.0)
    next_state = step(state, p, harvest=0.0)

    expected_algae = 10.0 * np.exp(p.alpha * p.dt)
    assert next_state.algae == pytest.approx(expected_algae, rel=1e-4)
    assert next_state.fish == 0.0


@pytest.mark.unit
def test_equilibrium_is_stationary() -> None:
    """At ``(F*, A*) = (alpha/beta, gamma/delta)``, state barely moves."""
    p = EcologyParams(dt=0.01, integrator="rk45")
    f_star = p.alpha / p.beta
    a_star = p.gamma / p.delta
    state = EcologicalState(fish=f_star, algae=a_star)

    next_state = step(state, p, harvest=0.0)

    assert abs(next_state.fish - f_star) < 1e-6
    assert abs(next_state.algae - a_star) < 1e-6


@pytest.mark.unit
def test_harvest_reduces_fish() -> None:
    p = EcologyParams(dt=0.1, integrator="rk45")
    state = EcologicalState(fish=10.0, algae=20.0)

    without_harvest = step(state, p, harvest=0.0)
    with_harvest = step(state, p, harvest=5.0)

    assert with_harvest.fish < without_harvest.fish


@pytest.mark.unit
def test_euler_unstable_with_extreme_harvest_raises() -> None:
    """Euler with absurd dt and harvest should produce negative biomass."""
    p = EcologyParams(integrator="euler", dt=10.0)
    state = EcologicalState(fish=10.0, algae=20.0)

    with pytest.raises(EcologyInstabilityError):
        step(state, p, harvest=1000.0)


@pytest.mark.unit
def test_rk45_positivity_preserved_for_reasonable_params() -> None:
    """RK45 should keep biomass positive over many steps at default params."""
    p = EcologyParams(dt=0.01, integrator="rk45")
    state = EcologicalState(fish=10.0, algae=20.0)

    for _ in range(1000):
        state = step(state, p, harvest=0.0)
        assert state.fish >= 0
        assert state.algae >= 0
