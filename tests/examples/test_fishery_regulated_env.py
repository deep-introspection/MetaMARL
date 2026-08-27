"""Deterministic tests for the fishery benchmark under the mechanism abstraction (TODO §3)."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.mechanism.algorithms.quota import QuotaMechanism
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.mechanism.composition.chained_mechanism import ChainedMechanism
from core.world.context import MechanismContext, MechanismStatus
from examples.bilevel_fishery.regulated_env import EPS, FisheryRegulatedEnv

AGENTS = ["u0", "u1"]
ECOLOGY = {
    "r": 0.3,
    "K": 1000.0,
    "p": 1.0,
    "fish_init": 800.0,
    "sigma": 0.0,
    "initial_stock_log_sigma": 0.0,
    "unregulated_f_multiplier": 2.0,
}


def resource_binding():
    return {"resource_level": lambda env: env.S_t["fish"] / max(env.K, EPS)}


def make_env(fake_world, mechanism, ecology=None, publish=True, **kw):
    env = FisheryRegulatedEnv(
        world=fake_world,
        mechanism_id=0,
        agents=AGENTS,
        mechanism=mechanism,
        ecology_cfg={**ECOLOGY, **(ecology or {})},
        action_spaces={
            aid: spaces.Box(-np.inf, np.inf, (2,), np.float32) for aid in AGENTS
        },
        seed=0,
        **kw,
    )
    ctx = (
        MechanismContext(
            index=0,
            env_id=None,
            seed=None,
            status=MechanismStatus.published,
            mechanism=mechanism,
            metrics=None,
        )
        if publish
        else None
    )
    fake_world.get_mechanism_by_id = SimpleNamespace(remote=lambda **k: ctx)
    return env


@pytest.mark.unit
def test_reference_points_schaefer():
    env = make_env(
        SimpleNamespace(), SubsidyMechanism(subsidy=0.0, cost=0.0), publish=False
    )
    # p = 1: B_msy = K/2, MSY = rK/4, F_msy = r/2
    assert env.B_msy == pytest.approx(500.0)
    assert env.MSY == pytest.approx(75.0)
    assert env.F_msy == pytest.approx(0.15)


@pytest.mark.unit
def test_reset_is_deterministic_and_clipped(fake_world):
    env = make_env(
        fake_world,
        SubsidyMechanism(subsidy=0.0, cost=0.0),
        ecology={"fish_init": 5000.0},
    )
    obs, _ = env.reset()
    assert env.S_t == {"fish": 1000.0, "last_usage": 0.0}
    np.testing.assert_allclose(obs["u0"][:2], [1.0, 0.0])


@pytest.mark.unit
def test_transition_matches_closed_form_without_quota(fake_world):
    env = make_env(fake_world, SubsidyMechanism(subsidy=0.0, cost=0.0))
    env.reset()
    fish = env.S_t["fish"]  # 800
    # raw 0 -> normalized 0.5 harvest fraction, 0.5 restoration effort (inert: effectiveness 0)
    obs, rewards, _, _, infos = env.step({aid: np.zeros(2) for aid in AGENTS})

    full = 2.0 * env.F_msy * fish / 2  # per agent
    H = 2 * 0.5 * full
    growth = 0.3 * fish * (1 - fish / 1000.0)
    expected_next = fish + growth - H
    assert env.S_t["fish"] == pytest.approx(expected_next)
    assert env.S_t["last_usage"] == pytest.approx(H)
    assert rewards == {
        "u0": pytest.approx(0.5),
        "u1": pytest.approx(0.5),
    }  # delivered fraction
    assert infos["u0"]["H_realized"] == pytest.approx(H)
    assert infos["u0"]["harvest_to_msy"] == pytest.approx(H / 75.0)
    assert infos["u0"]["restoration"] == 0.0
    assert infos["u0"]["delivered_harvest"] == pytest.approx(0.5 * full)
    np.testing.assert_allclose(obs["u0"][:2], [expected_next / 1000.0, H / 1000.0])


@pytest.mark.unit
def test_restoration_component_feeds_growth(fake_world):
    env = make_env(
        fake_world,
        SubsidyMechanism(subsidy=0.0, cost=0.0),
        ecology={"restoration_effectiveness": 10.0},
    )
    env.reset()
    fish = env.S_t["fish"]
    # harvest raw -50 -> ~0 fraction; restoration raw +50 -> ~1 effort each
    env.step({aid: np.array([-50.0, 50.0]) for aid in AGENTS})
    growth = 0.3 * fish * (1 - fish / 1000.0)
    assert env.S_t["fish"] == pytest.approx(fish + growth + 10.0 * 2.0, rel=1e-4)
    assert env._infos["u0"]["restoration"] == pytest.approx(20.0, rel=1e-4)


@pytest.mark.unit
def test_stock_is_bounded_by_zero_and_K(fake_world):
    env = make_env(
        fake_world,
        SubsidyMechanism(subsidy=0.0, cost=0.0),
        ecology={"unregulated_f_multiplier": 50.0},
    )
    env.reset()
    for _ in range(5):
        env.step({aid: np.array([50.0, -50.0]) for aid in AGENTS})
    assert 0.0 <= env.S_t["fish"] <= env.K
    env2 = make_env(
        fake_world,
        SubsidyMechanism(subsidy=0.0, cost=0.0),
        ecology={"restoration_effectiveness": 1e6},
    )
    env2.reset()
    env2.step({aid: np.array([-50.0, 50.0]) for aid in AGENTS})
    assert env2.S_t["fish"] == env2.K


@pytest.mark.unit
def test_quota_caps_harvest_when_stock_is_low(fake_world):
    quota = QuotaMechanism(fixed_quota=0.6, bindings=resource_binding())
    env = make_env(
        fake_world, quota, ecology={"fish_init": 100.0}
    )  # fish_norm 0.1 << quota
    env.reset()
    _, rewards, _, _, infos = env.step({aid: np.array([50.0, 0.0]) for aid in AGENTS})
    allowed = quota.allowed_fraction(0.1)
    assert allowed < 0.01
    assert rewards["u0"] == pytest.approx(allowed, abs=0.01)
    assert infos["u0"]["requested_frac"] == pytest.approx(
        allowed, abs=0.01
    )  # post-mechanism
    assert env.S_t["fish"] > 100.0  # stock recovers under the cap


@pytest.mark.unit
def test_subsidy_shapes_reward_without_touching_ecology(fake_world):
    chain = ChainedMechanism(
        children=(
            QuotaMechanism(fixed_quota=0.5, bindings=resource_binding()),
            SubsidyMechanism(subsidy=0.4, cost=0.1, action_component=1),
        )
    )
    env = make_env(fake_world, chain)
    env.reset()
    fish = env.S_t["fish"]
    _, rewards, *_ = env.step({aid: np.array([0.0, 50.0]) for aid in AGENTS})
    effort = 1.0
    assert rewards["u0"] == pytest.approx(
        0.5 + 0.4 * effort - 0.1 * effort**2, abs=1e-3
    )
    growth = 0.3 * fish * (1 - fish / 1000.0)
    full = 2.0 * env.F_msy * fish / 2
    assert env.S_t["fish"] == pytest.approx(fish + growth - 2 * 0.5 * full, rel=1e-4)
