"""Unit tests for ``MultiAgentRegulatedEnv`` with an in-memory World (TODO §2, §12, §13)."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from core.envs import hooks
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.mechanism.algorithms.quota import QuotaMechanism
from core.mechanism.algorithms.social_influence import SocialInfluenceMechanism
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.mechanism.composition.chained_mechanism import ChainedMechanism
from core.utils import sigmoid
from core.world.context import EnvStepContext, MechanismContext, MechanismStatus

AGENTS = ["a", "b"]
BOX2 = spaces.Box(-np.inf, np.inf, (2,), np.float32)


class StockEnv(MultiAgentRegulatedEnv):
    """Minimal benchmark: a stock depleted by the first action component."""

    @hooks.reset
    def init_state(self):
        return {"stock": 1.0}

    @hooks.reward
    def base_reward(self, A_t):
        return {aid: float(a[0]) for aid, a in A_t.items()}

    @hooks.transition
    def dynamics(self, *, A_t, S_t):
        return {
            "stock": max(S_t["stock"] - sum(float(a[0]) for a in A_t.values()), 0.0)
        }

    @hooks.observation
    def obs(self, observation_dict):
        return {
            aid: np.array([self.S_t["stock"]], dtype=np.float32) for aid in self.agents
        }


def resource_binding():
    return {"resource_level": lambda env: env.S_t["stock"]}


def social_bindings():
    return {
        "previous_actions": lambda env: env.previous_actions,
        "agent_ids": lambda env: tuple(env.agents),
    }


def make_env(fake_world, mechanism, **kw):
    kw.setdefault("action_spaces", {aid: BOX2 for aid in AGENTS})
    return StockEnv(
        world=fake_world,
        mechanism_id=0,
        agents=AGENTS,
        mechanism=mechanism,
        horizon=3,
        **kw,
    )


def publish(fake_world, env, mechanism):
    """Simulate the outer optimizer publishing a candidate for this env."""
    fake_world.get_mechanism_by_id = SimpleNamespace(
        remote=lambda **kw: MechanismContext(
            index=0,
            env_id=None,
            seed=env.policy_seed,
            status=MechanismStatus.published,
            mechanism=mechanism,
            metrics=None,
        )
    )


@pytest.fixture
def world(fake_world):
    # default: nothing published
    fake_world.get_mechanism_by_id = SimpleNamespace(remote=lambda **kw: None)
    return fake_world


@pytest.mark.unit
def test_action_temperature_validation(world):
    with pytest.raises(ValueError):
        make_env(world, SubsidyMechanism(subsidy=0.1, cost=0.1), action_temperature=0.0)


@pytest.mark.unit
def test_mechanism_property_requires_template_or_publication(world):
    env = make_env(world, mechanism=None)
    with pytest.raises(RuntimeError, match="No mechanism available"):
        _ = env.mechanism


@pytest.mark.unit
def test_action_pipeline_normalizes_then_applies_mechanism(world):
    quota = QuotaMechanism(fixed_quota=0.5, bindings=resource_binding())
    env = make_env(world, quota)
    env.reset()
    env.S_t = {"stock": 0.0}  # depleted -> allowance ~0
    out = env.action({"a": np.array([8.0, -8.0]), "b": np.array([0.0, 0.0])})
    # component 1 untouched by the quota, sigmoid-normalized
    assert out["a"][1] == pytest.approx(sigmoid(-8.0 / 4.0))
    assert out["b"][1] == pytest.approx(0.5)
    # component 0 capped near the (zero) allowance
    assert out["a"][0] < 0.05


@pytest.mark.unit
def test_inert_step_before_publication(world):
    env = make_env(world, SubsidyMechanism(subsidy=0.1, cost=0.1))
    obs, _ = env.reset()
    assert not env.published_mechanism_assigned
    obs2, rewards, term, trunc, infos = env.step({aid: np.zeros(2) for aid in AGENTS})
    assert rewards == {"a": 0.0, "b": 0.0}
    assert term["__all__"] is False and trunc["__all__"] is False
    assert obs2["a"].shape == obs["a"].shape
    assert world.contexts == []  # nothing published on inert steps
    assert env.S_t == {"stock": 1.0}  # no dynamics


@pytest.mark.unit
def test_full_step_pipeline_with_published_mechanism(world):
    subsidy = SubsidyMechanism(subsidy=0.5, cost=0.0, action_component=1)
    env = make_env(world, mechanism=subsidy, seed=7, policy_seed=11)
    publish(world, env, subsidy)
    obs, infos = env.reset()
    assert env.published_mechanism_assigned

    # obs = [stock] + theta (subsidy encode = 1.0)
    np.testing.assert_allclose(obs["a"], [1.0, 1.0])

    raw = {"a": np.array([0.0, 40.0]), "b": np.array([0.0, -40.0])}
    obs, rewards, term, trunc, infos = env.step(raw)

    # normalized: harvest 0.5 each, effort ~1 for a and ~0 for b
    assert env.S_t["stock"] == pytest.approx(0.0)
    assert rewards["a"] == pytest.approx(
        0.5 + 0.5 * 1.0, abs=1e-4
    )  # base + subsidy*effort
    assert rewards["b"] == pytest.approx(0.5 + 0.0, abs=1e-4)
    np.testing.assert_allclose(obs["a"], [0.0, 1.0])
    assert infos["a"]["intrinsic_utility"] == pytest.approx(0.5)
    assert np.allclose(env.previous_actions["a"], [0.5, sigmoid(10.0)])

    # exactly one EnvStepContext published with the regulated values
    assert len(world.contexts) == 1
    payload = world.contexts[0].payload
    assert isinstance(payload, EnvStepContext)
    assert payload.mechanism == 0 and payload.seed == 7 and payload.policy_seed == 11
    assert payload.status == MechanismStatus.train
    assert payload.reward == rewards
    np.testing.assert_allclose(payload.action["a"], env.previous_actions["a"])


@pytest.mark.unit
def test_horizon_truncates(world):
    subsidy = SubsidyMechanism(subsidy=0.1, cost=0.1)
    env = make_env(world, subsidy)
    publish(world, env, subsidy)
    env.reset()
    zeros = {aid: np.zeros(2) for aid in AGENTS}
    for expected in (False, False, True):
        *_, trunc, _ = env.step(zeros)
        assert trunc["__all__"] is expected
    assert env._t == 3
    env.reset()
    assert env._t == 0


@pytest.mark.unit
def test_observation_dimension_is_constant_and_matches_formula(world):
    """base + theta + quota allowed_frac + (N-1)*d social features (TODO §12)."""
    chain = ChainedMechanism(
        children=(
            QuotaMechanism(fixed_quota=0.5, bindings=resource_binding()),
            SubsidyMechanism(subsidy=0.1, cost=0.1),
            SocialInfluenceMechanism(bindings=social_bindings()),
        )
    )
    env = make_env(world, chain)
    publish(world, env, chain)
    obs, _ = env.reset()
    expected_dim = 1 + chain.to_vector().shape[0] + 1 + (len(AGENTS) - 1) * 2
    assert expected_dim == 1 + 2 + 1 + 2
    assert all(o.shape == (expected_dim,) for o in obs.values())
    obs, *_ = env.step({aid: np.zeros(2) for aid in AGENTS})
    assert all(o.shape == (expected_dim,) for o in obs.values())
    # social features of "a" are b's previous (normalized) actions = 0.5, 0.5
    np.testing.assert_allclose(obs["a"][-2:], [0.5, 0.5])


@pytest.mark.unit
def test_world_fetch_failure_is_wrapped(fake_world):
    def boom(**kw):
        raise ConnectionError("actor died")

    fake_world.get_mechanism_by_id = SimpleNamespace(remote=boom)
    env = make_env(fake_world, SubsidyMechanism(subsidy=0.1, cost=0.1))
    with pytest.raises(RuntimeError, match="Could not fetch mechanism_id=0"):
        env.reset()
