"""Tests for ``ThresholdPenaltyMechanism``."""

import numpy as np
import pytest

from core.mechanism.algorithms.penalty import ThresholdPenaltyMechanism


def make(**kw):
    kw.setdefault("bindings", {"resource_level": lambda env: env.S_t["fish"] / env.K})
    return ThresholdPenaltyMechanism(**kw)


@pytest.mark.unit
class TestValidation:
    def test_requires_binding(self):
        with pytest.raises(ValueError, match="resource_level"):
            ThresholdPenaltyMechanism()

    def test_ranges(self):
        with pytest.raises(ValueError):
            make(threshold=1.5)
        with pytest.raises(ValueError):
            make(penalty_amount=-1.0)
        with pytest.raises(ValueError):
            make(transition_width=0.0)


@pytest.mark.unit
class TestOptimizerSpace:
    def test_fixed_mechanism(self):
        m = make(threshold=0.2, penalty_amount=0.1)
        assert m.dimension == 0
        assert m.encode().shape == (0,)
        assert m.param_names() == []
        assert m.decode(np.empty(0)) is m
        assert m.clip() is m
        np.testing.assert_allclose(m.to_vector(), [0.2, 0.1])

    def test_decode_rejects_non_empty(self):
        with pytest.raises(ValueError):
            make().decode(np.array([0.1]))


@pytest.mark.unit
class TestRewardChannel:
    def test_far_above_threshold_no_penalty(self, env_at):
        m = make(threshold=0.2, penalty_amount=0.1, transition_width=0.03)
        out = m.reward({"a": 1.0, "b": 2.0}, **m.resolve(env_at(fish_norm=0.9)))
        assert out["a"] == pytest.approx(1.0, abs=1e-6)
        assert out["b"] == pytest.approx(2.0, abs=1e-6)

    def test_far_below_threshold_full_penalty(self, env_at):
        m = make(threshold=0.2, penalty_amount=0.1, transition_width=0.03)
        out = m.reward({"a": 1.0}, **m.resolve(env_at(fish_norm=0.0)))
        assert out["a"] == pytest.approx(0.9, abs=1e-3)

    def test_half_penalty_at_threshold(self):
        m = make(threshold=0.2, penalty_amount=0.1)
        assert m.penalty(0.2) == pytest.approx(0.05)

    def test_extreme_inputs_do_not_overflow(self):
        m = make(threshold=0.5, penalty_amount=1.0, transition_width=1e-4)
        assert np.isfinite(m.penalty(0.0))
        assert np.isfinite(m.penalty(1.0))
