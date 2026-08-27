"""Numerical tests for ``QuotaMechanism`` (TODO §6)."""

import numpy as np
import pytest

from core.mechanism.algorithms.quota import QuotaMechanism
from core.utils import sigmoid


def make_quota(q=0.5, width=0.03, usage_width=0.005, **kw):
    return QuotaMechanism(
        fixed_quota=q,
        bindings={"resource_level": lambda env: env.S_t["fish"] / env.K},
        quota_transition_width=width,
        usage_transition_width=usage_width,
        **kw,
    )


@pytest.mark.unit
class TestValidation:
    def test_requires_resource_level_binding(self):
        with pytest.raises(ValueError, match="resource_level"):
            QuotaMechanism(fixed_quota=0.5, bindings={})

    @pytest.mark.parametrize("q", [-0.1, 1.1])
    def test_quota_out_of_range(self, q):
        with pytest.raises(ValueError, match="fixed_quota"):
            make_quota(q=q)

    def test_non_positive_widths(self):
        with pytest.raises(ValueError):
            make_quota(width=0.0)
        with pytest.raises(ValueError):
            make_quota(usage_width=-1.0)


@pytest.mark.unit
class TestOptimizerSpace:
    def test_dimension_and_names(self):
        m = make_quota(q=0.3)
        assert m.dimension == 1
        assert m.param_names() == ["fixed_quota"]
        np.testing.assert_allclose(m.encode(), [0.3])
        np.testing.assert_allclose(m.to_vector(), [0.3])

    def test_decode_round_trip_and_immutability(self):
        m = make_quota(q=0.3)
        m2 = m.decode(np.array([0.7], dtype=np.float32))
        assert m2 is not m
        assert m.fixed_quota == 0.3
        assert m2.fixed_quota == pytest.approx(0.7)
        np.testing.assert_allclose(m2.decode(m2.encode()).encode(), m2.encode())

    def test_decode_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            make_quota().decode(np.array([0.1, 0.2]))

    def test_clip(self):
        m = make_quota(q=0.5)
        object.__setattr__(m, "fixed_quota", 1.4)  # bypass validation on purpose
        assert m.clip().fixed_quota == 1.0


@pytest.mark.unit
class TestAllowedFraction:
    def test_extremes(self):
        m = make_quota(q=0.5, width=0.03)
        assert m.allowed_fraction(0.0) == pytest.approx(0.0, abs=1e-6)
        assert m.allowed_fraction(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_transition_at_quota(self):
        q, w = 0.4, 0.05
        m = make_quota(q=q, width=w)
        lower, upper = sigmoid(-q / w), sigmoid((1 - q) / w)
        expected = (0.5 - lower) / (upper - lower)
        assert m.allowed_fraction(q) == pytest.approx(expected, rel=1e-6)
        assert m.allowed_fraction(q - 3 * w) < 0.1
        assert m.allowed_fraction(q + 3 * w) > 0.9

    def test_monotonic_in_resource(self):
        m = make_quota()
        levels = np.linspace(0, 1, 50)
        values = [m.allowed_fraction(b) for b in levels]
        assert np.all(np.diff(values) >= 0)


@pytest.mark.unit
class TestActionChannel:
    def test_request_below_allowance_unchanged(self, env_at):
        m = make_quota(q=0.5)
        env = env_at(fish_norm=0.9)  # allowance ~1
        actions = {"a": np.array([0.3, 0.7], dtype=np.float32)}
        out = m.action(actions, **m.resolve(env))
        np.testing.assert_allclose(out["a"], [0.3, 0.7], atol=1e-5)

    def test_request_above_allowance_is_capped(self, env_at):
        m = make_quota(q=0.5, usage_width=0.005)
        env = env_at(fish_norm=0.05)  # allowance ~0
        allowed = m.allowed_fraction(0.05)
        actions = {"a": np.array([0.9, 0.7], dtype=np.float32)}
        out = m.action(actions, **m.resolve(env))
        assert out["a"][0] == pytest.approx(allowed, abs=0.01)
        assert out["a"][0] < 0.9

    def test_non_target_component_and_inputs_untouched(self, env_at):
        m = make_quota(q=0.5, action_component=0)
        env = env_at(fish_norm=0.05)
        original = np.array([0.9, 0.7], dtype=np.float32)
        actions = {"a": original, "b": original.copy()}
        out = m.action(actions, **m.resolve(env))
        np.testing.assert_allclose(original, [0.9, 0.7])  # not mutated in place
        assert out["a"][1] == pytest.approx(0.7)
        assert set(out) == {"a", "b"}

    def test_smooth_cap_is_continuous(self, env_at):
        m = make_quota(q=0.5)
        env = env_at(fish_norm=0.5)
        allowed = m.allowed_fraction(0.5)
        requests = np.linspace(allowed - 0.05, allowed + 0.05, 200)
        delivered = [
            m.action({"a": np.array([r, 0.0])}, **m.resolve(env))["a"][0]
            for r in requests
        ]
        assert np.max(np.abs(np.diff(delivered))) < 0.01
        assert np.all(np.diff(delivered) >= -1e-6)


@pytest.mark.unit
class TestObservationChannel:
    def test_no_context_is_identity(self):
        m = make_quota()
        obs = {"a": np.zeros(2, dtype=np.float32)}
        assert m.observation(obs) is obs

    def test_appends_allowed_frac_after_action(self, env_at):
        m = make_quota(q=0.5)
        env = env_at(fish_norm=0.5)
        m.action({"a": np.array([0.2, 0.2])}, **m.resolve(env))
        out = m.observation({"a": np.zeros(2, dtype=np.float32)})
        assert out["a"].shape == (3,)
        assert out["a"][2] == pytest.approx(m.allowed_fraction(0.5))
        assert m.observation_names() == ["effective_quota"]
