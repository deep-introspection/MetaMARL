"""API conformance and behavioural tests for FisheryEnv."""

import numpy as np
import pytest

from bilevel_fishery.ecology import EcologyParams
from bilevel_fishery.envs import FisheryEnv


@pytest.mark.unit
def test_reset_returns_obs_and_info() -> None:
    env = FisheryEnv()
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


@pytest.mark.unit
def test_reset_seeded_reproducible() -> None:
    obs1, _ = FisheryEnv().reset(seed=42)
    obs2, _ = FisheryEnv().reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)


@pytest.mark.unit
def test_reset_different_seeds_differ() -> None:
    p = EcologyParams(noise_std=0.05)  # noise enables variation
    obs1, _ = FisheryEnv(params=p).reset(seed=1)
    obs2, _ = FisheryEnv(params=p).reset(seed=2)
    assert not np.array_equal(obs1, obs2)


@pytest.mark.unit
def test_step_returns_five_tuple() -> None:
    env = FisheryEnv()
    env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert set(info) >= {"harvest_demanded", "harvest_realized", "fish", "algae"}


@pytest.mark.unit
def test_step_without_reset_raises() -> None:
    env = FisheryEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([0.5], dtype=np.float32))


@pytest.mark.unit
def test_horizon_truncates_episode() -> None:
    env = FisheryEnv(horizon=5)
    env.reset(seed=42)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, _ = env.step(np.array([0.5], dtype=np.float32))
    assert not terminated
    assert truncated


@pytest.mark.unit
def test_zero_action_yields_zero_reward() -> None:
    env = FisheryEnv()
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert reward == 0.0
    assert info["harvest_realized"] == 0.0


@pytest.mark.unit
def test_physical_cap_engages_when_stock_low() -> None:
    """When the stock is tiny, realized < demanded."""
    p = EcologyParams(fish_init=0.05, noise_std=0.0)
    env = FisheryEnv(params=p, max_harvest_rate=10.0)
    env.reset(seed=42)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["harvest_realized"] < info["harvest_demanded"]


@pytest.mark.unit
def test_terminated_is_always_false() -> None:
    env = FisheryEnv(horizon=20)
    env.reset(seed=42)
    for _ in range(20):
        _, _, terminated, _, _ = env.step(np.array([0.7], dtype=np.float32))
        assert not terminated


@pytest.mark.unit
def test_reward_is_concave_log1p() -> None:
    """``reward == log(1 + harvest_realized)``."""
    env = FisheryEnv()
    env.reset(seed=42)
    _, reward, _, _, info = env.step(np.array([0.3], dtype=np.float32))
    assert reward == pytest.approx(float(np.log1p(info["harvest_realized"])))


@pytest.mark.unit
def test_extreme_overharvest_collapses_gracefully() -> None:
    """Sustained max action with low initial stock should not crash the env.

    Even if `ecology.step` raises `EcologyInstabilityError` because the cap
    heuristic missed by a numerical hair, the env layer must keep advancing
    so the agent simply sees ``harvest_realized = 0`` and stock at 0.
    """
    p = EcologyParams(fish_init=0.05, algae_init=2.0, noise_std=0.0, dt=0.05)
    env = FisheryEnv(params=p, max_harvest_rate=10.0, horizon=200)
    env.reset(seed=42)
    for _ in range(200):
        _, reward, _, truncated, info = env.step(np.array([1.0], dtype=np.float32))
        assert info["fish"] >= 0.0
        assert reward >= 0.0
        if truncated:
            break


@pytest.mark.unit
def test_observation_bounds_respected() -> None:
    env = FisheryEnv(horizon=100)
    env.reset(seed=42)
    for _ in range(100):
        obs, _, _, truncated, _ = env.step(np.array([0.4], dtype=np.float32))
        assert obs.dtype == np.float32
        assert obs.shape == (2,)
        assert (obs >= 0.0).all()
        assert (obs <= 1.0).all()
        if truncated:
            break
