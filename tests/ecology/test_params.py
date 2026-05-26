"""Validation tests for EcologyParams."""

import pytest
from pydantic import ValidationError

from bilevel_fishery.ecology.params import EcologyParams


@pytest.mark.unit
def test_default_params_validate() -> None:
    p = EcologyParams()
    assert p.alpha == 1.0
    assert p.beta == 0.1
    assert p.delta == 0.075
    assert p.gamma == 1.5
    assert p.integrator == "rk45"


@pytest.mark.unit
def test_default_equilibrium_is_consistent() -> None:
    """At default params, F* = alpha/beta = 10 and A* = gamma/delta = 20."""
    p = EcologyParams()
    assert p.alpha / p.beta == pytest.approx(10.0)
    assert p.gamma / p.delta == pytest.approx(20.0)


@pytest.mark.unit
def test_negative_alpha_rejected() -> None:
    with pytest.raises(ValidationError):
        EcologyParams(alpha=-1.0)


@pytest.mark.unit
def test_zero_dt_rejected() -> None:
    with pytest.raises(ValidationError):
        EcologyParams(dt=0.0)


@pytest.mark.unit
def test_init_exceeding_max_rejected() -> None:
    with pytest.raises(ValidationError):
        EcologyParams(fish_init=200.0, max_fish=100.0)


@pytest.mark.unit
def test_invalid_integrator_rejected() -> None:
    with pytest.raises(ValidationError):
        EcologyParams(integrator="midpoint")  # type: ignore[arg-type]


@pytest.mark.unit
def test_params_are_frozen() -> None:
    """EcologyParams instances should be immutable."""
    p = EcologyParams()
    with pytest.raises(ValidationError):
        p.alpha = 2.0  # type: ignore[misc]
