"""Analytical tests for ``SubsidyMechanism`` (TODO §4)."""

import numpy as np
import pytest

from core.mechanism.algorithms.subsidy import MAX_SUBSIDY, SubsidyMechanism


@pytest.mark.unit
class TestValidation:
    def test_bounds(self):
        with pytest.raises(ValueError, match="subsidy"):
            SubsidyMechanism(subsidy=0.6, cost=0.1)
        with pytest.raises(ValueError, match="cost"):
            SubsidyMechanism(subsidy=0.1, cost=1.5)


@pytest.mark.unit
class TestOptimizerSpace:
    def test_encode_normalizes_by_max(self):
        m = SubsidyMechanism(subsidy=0.25, cost=0.1)
        assert m.dimension == 1
        assert m.param_names() == ["restoration_subsidy"]
        np.testing.assert_allclose(m.encode(), [0.25 / MAX_SUBSIDY])
        np.testing.assert_allclose(m.to_vector(), m.encode())

    def test_decode_denormalizes(self):
        m = SubsidyMechanism(subsidy=0.1, cost=0.2).decode(np.array([1.0]))
        assert m.subsidy == pytest.approx(MAX_SUBSIDY)
        assert m.cost == 0.2  # fixed parameter preserved

    def test_clip(self):
        m = SubsidyMechanism(subsidy=0.1, cost=0.2)
        object.__setattr__(m, "subsidy", 0.9)
        assert m.clip().subsidy == MAX_SUBSIDY


@pytest.mark.unit
class TestRewardChannel:
    def test_zero_effort_no_change(self):
        m = SubsidyMechanism(subsidy=0.3, cost=0.5, action_component=1)
        out = m.reward({"a": 2.0}, action_after={"a": np.array([0.8, 0.0])})
        assert out["a"] == pytest.approx(2.0)
        assert isinstance(out["a"], float)

    def test_analytical_value(self):
        sigma, c, e = 0.3, 0.5, 0.4
        m = SubsidyMechanism(subsidy=sigma, cost=c, action_component=1)
        out = m.reward({"a": 1.0}, action_after={"a": np.array([0.8, e])})
        assert out["a"] == pytest.approx(1.0 + sigma * e - c * e**2)

    def test_component_selection(self):
        m = SubsidyMechanism(subsidy=0.5, cost=0.0, action_component=0)
        out = m.reward({"a": 0.0}, action_after={"a": np.array([0.2, 0.9])})
        assert out["a"] == pytest.approx(0.5 * 0.2)

    def test_per_agent_mapping(self):
        m = SubsidyMechanism(subsidy=0.5, cost=0.0)
        out = m.reward(
            {"a": 0.0, "b": 1.0},
            action_after={"a": np.array([0.0, 0.2]), "b": np.array([0.0, 1.0])},
        )
        assert out["a"] == pytest.approx(0.1)
        assert out["b"] == pytest.approx(1.5)
