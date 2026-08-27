"""Tests for ``SocialInfluenceMechanism`` (TODO §5)."""

import numpy as np
import pytest

from core.mechanism.algorithms.social_influence import SocialInfluenceMechanism

BINDINGS = {
    "previous_actions": lambda env: env.previous_actions,
    "agent_ids": lambda env: tuple(env.agents),
}


@pytest.mark.unit
def test_requires_bindings():
    with pytest.raises(ValueError, match="agent_ids"):
        SocialInfluenceMechanism(bindings={"previous_actions": lambda e: {}})


@pytest.mark.unit
def test_fixed_mechanism_api():
    m = SocialInfluenceMechanism(bindings=BINDINGS)
    assert m.dimension == 0
    assert m.encode().shape == (0,)
    assert m.to_vector().shape == (0,)
    assert m.param_names() == []
    assert m.decode(np.empty(0)) is m
    assert m.clip() is m
    assert m.influence_weight == 0.0  # reserved for the KL bonus, unused


@pytest.mark.unit
def test_peer_ordering_and_self_exclusion(env_at):
    m = SocialInfluenceMechanism(bindings=BINDINGS)
    env = env_at(
        fish_norm=0.5,
        agents=["a", "b", "c"],
        previous_actions={
            "a": np.array([1.0, 10.0]),
            "b": np.array([2.0, 20.0]),
            "c": np.array([3.0, 30.0]),
        },
    )
    obs = {aid: np.array([0.5, 0.1], dtype=np.float32) for aid in env.agents}
    out = m.observation(obs, **m.resolve(env))

    # (N - 1) * d = 2 * 2 = 4 extra features
    assert all(v.shape == (6,) for v in out.values())
    np.testing.assert_allclose(out["a"], [0.5, 0.1, 2.0, 20.0, 3.0, 30.0])
    np.testing.assert_allclose(out["b"], [0.5, 0.1, 1.0, 10.0, 3.0, 30.0])
    np.testing.assert_allclose(out["c"], [0.5, 0.1, 1.0, 10.0, 2.0, 20.0])
    assert all(v.dtype == np.float32 for v in out.values())
